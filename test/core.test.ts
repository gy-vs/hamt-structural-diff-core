import { describe, expect, it } from 'vitest';
import {
  PersistentMap,
  compareByHashPath,
  diffMaps,
  hashKey,
  type DiffChange,
} from '../src/index.js';

it('persists', () => {
  const a = new PersistentMap<number>();
  const b = a.set('x', 1);
  expect(a.size()).toBe(0);
  expect(b.get('x')).toBe(1);
});

async function collect<V>(changes: AsyncIterable<DiffChange<V>>): Promise<DiffChange<V>[]> {
  const out: DiffChange<V>[] = [];
  for await (const change of changes) out.push(change);
  return out;
}

function mulberry32(seed: number): () => number {
  let state = seed;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Order-independent reference diff computed from plain entry lists. */
function referenceDiff(a: PersistentMap<number>, b: PersistentMap<number>): DiffChange<number>[] {
  const changes: DiffChange<number>[] = [];
  const aItems = new Map(a.items().map((entry) => [entry.key, entry] as const));
  const bItems = new Map(b.items().map((entry) => [entry.key, entry] as const));
  for (const [key, entry] of aItems) {
    const other = bItems.get(key);
    if (!other) changes.push({ type: 'removed', key, hash: entry.hash, value: entry.value });
    else if (!Object.is(entry.value, other.value)) {
      changes.push({ type: 'changed', key, hash: entry.hash, oldValue: entry.value, newValue: other.value });
    }
  }
  for (const [key, entry] of bItems) {
    if (!aItems.has(key)) changes.push({ type: 'added', key, hash: entry.hash, value: entry.value });
  }
  changes.sort((x, y) => compareByHashPath(x.hash, x.key, y.hash, y.key));
  return changes;
}

function expectSortedByHashPath(changes: DiffChange<unknown>[]): void {
  for (let i = 1; i < changes.length; i++) {
    const prev = changes[i - 1];
    const next = changes[i];
    expect(compareByHashPath(prev.hash, prev.key, next.hash, next.key)).toBeLessThan(0);
  }
}

describe('hamt core', () => {
  it('supports basic persistent operations across node shapes', () => {
    let map = new PersistentMap<number>();
    expect(map.size()).toBe(0);
    expect(map.rootKind).toBe('empty');
    expect(map.get('missing')).toBeUndefined();

    for (let i = 0; i < 1000; i++) map = map.set('k' + i, i);
    expect(map.size()).toBe(1000);
    expect(map.get('k999')).toBe(999);
    expect(map.has('k500')).toBe(true);

    const deleted = map.delete('k500');
    expect(deleted.size()).toBe(999);
    expect(deleted.has('k500')).toBe(false);
    expect(map.has('k500')).toBe(true); // original untouched
    expect(map.set('k500', -500).get('k500')).toBe(-500);
    expect(map.delete('nope')).toBe(map); // no-op keeps identity

    // Collision node operations.
    let coll = new PersistentMap<number>();
    for (let i = 0; i < 50; i++) coll = coll.set('c' + i, i, 0xbeef);
    expect(coll.rootKind).toBe('collision');
    expect(coll.size()).toBe(50);
    expect(coll.get('c25', 0xbeef)).toBe(25);
    coll = coll.delete('c25', 0xbeef);
    expect(coll.get('c25', 0xbeef)).toBeUndefined();
    expect(coll.size()).toBe(49);

    // Deleting collapses single-terminal bitmap nodes back to a leaf.
    let deep = new PersistentMap<number>().set('a', 1, 0).set('b', 2, 32);
    expect(deep.rootKind).toBe('bitmap');
    deep = deep.delete('b', 32);
    expect(deep.rootKind).toBe('leaf');
    expect(deep.get('a', 0)).toBe(1);
  });

  it('converts between bitmap and array nodes as density changes', () => {
    let map = new PersistentMap<number>();
    for (let i = 0; i < 17; i++) map = map.set('k' + i, i, i);
    expect(map.rootKind).toBe('array');
    for (let i = 16; i >= 5; i--) map = map.delete('k' + i, i);
    expect(map.rootKind).toBe('bitmap');
    expect(map.size()).toBe(5);
    for (let i = 0; i < 5; i++) expect(map.get('k' + i, i)).toBe(i);
  });
});

describe('diff', () => {
  it('returns an empty diff for identical references without visiting any node', async () => {
    let a = new PersistentMap<number>();
    for (let i = 0; i < 1000; i++) a = a.set('key-' + i, i);

    const stats = { nodesVisited: 0 };
    expect(await collect(a.diff(a, { stats }))).toEqual([]);
    expect(stats.nodesVisited).toBe(0);

    // A no-op set returns the same reference, so the diff is still free.
    const b = a.set('key-1', a.get('key-1')!);
    expect(b).toBe(a);
    const stats2 = { nodesVisited: 0 };
    expect(await collect(a.diff(b, { stats: stats2 }))).toEqual([]);
    expect(stats2.nodesVisited).toBe(0);
  });

  it('returns an empty diff for logically equal maps with no shared nodes', async () => {
    const keys = Array.from({ length: 500 }, (_, i) => 'key-' + i);
    let a = new PersistentMap<number>();
    for (const key of keys) a = a.set(key, hashKey(key));
    // Built independently, in reverse order: no shared node references.
    let b = new PersistentMap<number>();
    for (const key of [...keys].reverse()) b = b.set(key, hashKey(key));

    expect(await collect(a.diff(b))).toEqual([]);

    const changed = b.set('key-1', -1).set('key-2', -2);
    const changes = await collect(a.diff(changed));
    expect(changes).toHaveLength(2);
    expect(changes.every((change) => change.type === 'changed')).toBe(true);
  });

  it('compares heterogeneous node shapes at the same logical position', async () => {
    const leafOnly = new PersistentMap<number>().set('solo', 1, 0x111);
    expect(leafOnly.rootKind).toBe('leaf');

    const withCollision = leafOnly.set('twin', 2, 0x111);
    expect(withCollision.rootKind).toBe('collision');

    const withBitmap = withCollision.set('other', 3, 0x222);
    expect(withBitmap.rootKind).toBe('bitmap');

    // leaf vs collision
    expect(await collect(leafOnly.diff(withCollision))).toEqual([
      { type: 'added', key: 'twin', hash: 0x111, value: 2 },
    ]);
    // collision vs bitmap
    expect(await collect(withCollision.diff(withBitmap))).toEqual([
      { type: 'added', key: 'other', hash: 0x222, value: 3 },
    ]);
    // leaf vs bitmap with a collision nested inside; 0x222 sorts before 0x111
    // because fragment(0x222) = 2 < fragment(0x111) = 17.
    expect(await collect(leafOnly.diff(withBitmap))).toEqual([
      { type: 'added', key: 'other', hash: 0x222, value: 3 },
      { type: 'added', key: 'twin', hash: 0x111, value: 2 },
    ]);
    // reverse direction
    expect(await collect(withBitmap.diff(leafOnly))).toEqual([
      { type: 'removed', key: 'other', hash: 0x222, value: 3 },
      { type: 'removed', key: 'twin', hash: 0x111, value: 2 },
    ]);
  });

  it('aligns bitmap and array nodes by hash fragment', async () => {
    let sparse = new PersistentMap<number>();
    for (let i = 0; i < 16; i++) sparse = sparse.set('key-' + i, i, i);
    expect(sparse.rootKind).toBe('bitmap');

    const dense = sparse.set('key-16', 16, 16);
    expect(dense.rootKind).toBe('array');

    expect(await collect(sparse.diff(dense))).toEqual([
      { type: 'added', key: 'key-16', hash: 16, value: 16 },
    ]);
    expect(await collect(dense.diff(sparse))).toEqual([
      { type: 'removed', key: 'key-16', hash: 16, value: 16 },
    ]);

    // A value change plus a shape change across the same boundary.
    const denseChanged = dense.set('key-3', -3, 3);
    expect(await collect(sparse.diff(denseChanged))).toEqual([
      { type: 'changed', key: 'key-3', hash: 3, oldValue: 3, newValue: -3 },
      { type: 'added', key: 'key-16', hash: 16, value: 16 },
    ]);
  });

  it('diffs large collision sets by key', async () => {
    const HASH = 0xdeadbeef;
    let a = new PersistentMap<number>();
    for (let i = 0; i < 2000; i++) a = a.set('k' + String(i).padStart(4, '0'), i, HASH);
    expect(a.rootKind).toBe('collision');

    const b = a
      .delete('k0001', HASH)
      .delete('k0002', HASH)
      .delete('k0003', HASH)
      .set('k1000', -1000, HASH)
      .set('k2000', 2000, HASH)
      .set('k2001', 2001, HASH);
    expect(b.rootKind).toBe('collision');

    const stats = { nodesVisited: 0 };
    const changes = await collect(a.diff(b, { stats }));
    expect(changes).toEqual([
      { type: 'removed', key: 'k0001', hash: HASH, value: 1 },
      { type: 'removed', key: 'k0002', hash: HASH, value: 2 },
      { type: 'removed', key: 'k0003', hash: HASH, value: 3 },
      { type: 'changed', key: 'k1000', hash: HASH, oldValue: 1000, newValue: -1000 },
      { type: 'added', key: 'k2000', hash: HASH, value: 2000 },
      { type: 'added', key: 'k2001', hash: HASH, value: 2001 },
    ]);
    // A single collision-node pair was compared; no trie walk happened.
    expect(stats.nodesVisited).toBe(1);

    // Collision vs internal node: adding an outside key wraps the collision in a bitmap.
    const wider = b.set('outside', 42, 7);
    expect(wider.rootKind).toBe('bitmap');
    const stats2 = { nodesVisited: 0 };
    expect(await collect(b.diff(wider, { stats: stats2 }))).toEqual([
      { type: 'added', key: 'outside', hash: 7, value: 42 },
    ]);
    expect(stats2.nodesVisited).toBeLessThan(10);
  });

  it('reports delete-then-add as a single changed entry', async () => {
    let base = new PersistentMap<number>();
    for (let i = 0; i < 200; i++) base = base.set('key-' + i, i);

    const removed = base.delete('key-42');

    // Re-added with the same value: no diff at all.
    const same = removed.set('key-42', 42);
    expect(await collect(base.diff(same))).toEqual([]);

    // Re-added with a different value: one `changed`, comparator sees the key once.
    const different = removed.set('key-42', -42);
    const calls: string[] = [];
    const changes = await collect(
      base.diff(different, {
        valueEquals: (x, y, key) => {
          calls.push(key);
          return Object.is(x, y);
        },
      }),
    );
    expect(changes).toEqual([
      { type: 'changed', key: 'key-42', hash: hashKey('key-42'), oldValue: 42, newValue: -42 },
    ]);
    expect(calls.filter((key) => key === 'key-42')).toHaveLength(1);
  });

  it('uses the injected comparator only for keys present on both sides', async () => {
    type Val = { id: number; tag: string };
    let a = new PersistentMap<Val>();
    let b = new PersistentMap<Val>();
    for (let i = 0; i < 100; i++) a = a.set('common-' + i, { id: i, tag: 'a' });
    for (let i = 0; i < 100; i++) b = b.set('common-' + i, { id: i, tag: 'b' });
    a = a.set('only-a', { id: -1, tag: 'a' });
    b = b.set('only-b', { id: -2, tag: 'b' });

    // Default Object.is: every shared key reports changed (distinct object identities).
    const strict = await collect(a.diff(b));
    expect(strict.filter((change) => change.type === 'changed')).toHaveLength(100);
    expect(strict.filter((change) => change.type === 'added')).toHaveLength(1);
    expect(strict.filter((change) => change.type === 'removed')).toHaveLength(1);

    // Custom equality on `id`: the 100 shared keys no longer report `changed`,
    // but the one-sided keys are still added/removed.
    const calls: string[] = [];
    const relaxed = await collect(
      a.diff(b, {
        valueEquals: (x, y, key) => {
          calls.push(key);
          return x.id === y.id;
        },
      }),
    );
    expect(relaxed).toEqual(
      [
        { type: 'removed', key: 'only-a', hash: hashKey('only-a'), value: { id: -1, tag: 'a' } },
        { type: 'added', key: 'only-b', hash: hashKey('only-b'), value: { id: -2, tag: 'b' } },
      ].sort((x, y) => compareByHashPath(x.hash, x.key, y.hash, y.key)),
    );
    expect(calls).toHaveLength(100);
    expect(new Set(calls).size).toBe(100);
    expect(calls).not.toContain('only-a');
    expect(calls).not.toContain('only-b');
  });

  it('matches a brute-force diff and stays sorted by hash path then key', async () => {
    const rand = mulberry32(12345);
    const pick = (n: number) => Math.floor(rand() * n);
    const strategies: Array<(key: string) => number> = [
      hashKey,
      (key) => Number(key.slice(1)) % 300, // crowded hash space: collisions + dense nodes
    ];

    for (const hashOf of strategies) {
      let a = new PersistentMap<number>();
      for (let i = 0; i < 400; i++) {
        const key = 'k' + pick(600);
        a = a.set(key, i, hashOf(key));
      }
      let b = a;
      for (let i = 0; i < 150; i++) {
        const key = 'k' + pick(600);
        b = pick(3) === 0 ? b.delete(key, hashOf(key)) : b.set(key, 1000 + i, hashOf(key));
      }

      const changes = await collect(a.diff(b));
      expect(changes).toEqual(referenceDiff(a, b));
      expectSortedByHashPath(changes);
    }
  });

  it('diffs empty maps and supports the standalone diffMaps API', async () => {
    const empty = new PersistentMap<number>();
    const full = empty.set('a', 1).set('b', 2);
    expect(await collect(empty.diff(empty))).toEqual([]);

    const added = [
      { type: 'added', key: 'a', hash: hashKey('a'), value: 1 },
      { type: 'added', key: 'b', hash: hashKey('b'), value: 2 },
    ] as const;
    const sorted = [...added].sort((x, y) => compareByHashPath(x.hash, x.key, y.hash, y.key));
    expect(await collect(diffMaps(empty, full))).toEqual(sorted);
    expect(await collect(diffMaps(full, empty))).toEqual(
      sorted.map(({ type, ...rest }) => ({ ...rest, type: 'removed' })),
    );
  });

  it('throws the abort reason when the signal is already aborted', async () => {
    const a = new PersistentMap<number>().set('x', 1);
    const b = a.set('y', 2);
    const error = await a
      .diff(b, { signal: AbortSignal.abort() })
      .next()
      .then(
        () => null,
        (err) => err,
      );
    expect(error).toBeInstanceOf(DOMException);
    expect((error as DOMException).name).toBe('AbortError');
  });

  it('stops mid-iteration when the signal aborts', async () => {
    let a = new PersistentMap<number>();
    for (let i = 0; i < 1000; i++) a = a.set('key-' + i, i);
    let b = a;
    for (let i = 0; i < 1000; i += 3) b = b.set('key-' + i, -i);

    const controller = new AbortController();
    const seen: DiffChange<number>[] = [];
    const error = await (async () => {
      try {
        for await (const change of a.diff(b, { signal: controller.signal })) {
          seen.push(change);
          if (seen.length === 3) controller.abort();
        }
        return null;
      } catch (err) {
        return err;
      }
    })();
    expect(seen).toHaveLength(3);
    expect((error as DOMException).name).toBe('AbortError');
  });

  it('stops walking the trie when the consumer breaks out of the loop', async () => {
    let a = new PersistentMap<number>();
    for (let i = 0; i < 1000; i++) a = a.set('key-' + i, i);
    let b = a;
    for (let i = 0; i < 1000; i += 3) b = b.set('key-' + i, -i);

    const stats = { nodesVisited: 0 };
    let count = 0;
    for await (const change of a.diff(b, { stats })) {
      void change;
      count++;
      break;
    }
    expect(count).toBe(1);

    const fullStats = { nodesVisited: 0 };
    const all = await collect(a.diff(b, { stats: fullStats }));
    expect(all.length).toBeGreaterThan(1);
    expect(stats.nodesVisited).toBeLessThan(fullStats.nodesVisited);
  });

  it('skips shared subtrees for small changes', async () => {
    let a = new PersistentMap<number>();
    for (let i = 0; i < 20000; i++) a = a.set('key-' + i, i);
    const total = a.nodeCount();
    expect(total).toBeGreaterThan(20000);

    const b = a.set('key-7', -7);
    const stats = { nodesVisited: 0 };
    const changes = await collect(a.diff(b, { stats }));
    expect(changes).toEqual([
      { type: 'changed', key: 'key-7', hash: hashKey('key-7'), oldValue: 7, newValue: -7 },
    ]);
    // Only the changed root-to-leaf path was compared, not the 20k-entry trie.
    expect(stats.nodesVisited).toBeLessThan(100);
    expect(stats.nodesVisited).toBeLessThanOrEqual(Math.ceil(total / 100));
  });
});
