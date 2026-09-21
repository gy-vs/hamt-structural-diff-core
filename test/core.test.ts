import { expect, it, describe } from 'vitest';
import {
  PersistentMap,
  diffMaps,
  diffIterate,
  hashKey,
  fragmentAt,
  type HAMTNode,
  type DiffEvent,
  type Entry,
  ArrayNode,
  BitmapNode,
  CollisionNode,
  LeafNode,
} from '../src/index.js';

/** Build a map by folding [key, value, hash?] tuples. */
function build<V>(pairs: Array<[string, V] | [string, V, number]>): PersistentMap<V> {
  let map = new PersistentMap<V>();
  for (const [key, value, hash] of pairs) map = map.set(key, value, hash ?? hashKey(key));
  return map;
}

/** Rebuild the same logical content from scratch (no shared references). */
function rebuild<V>(map: PersistentMap<V>): PersistentMap<V> {
  let copy = new PersistentMap<V>();
  for (const entry of map.items()) copy = copy.set(entry.key, entry.value, entry.hash);
  return copy;
}

function kindsOf(map: PersistentMap<unknown>): Set<string> {
  const kinds = new Set<string>();
  const walk = (node: HAMTNode<unknown> | null): void => {
    if (!node) return;
    kinds.add(node.kind);
    if (node.kind === 'bitmap') node.children.forEach(walk);
    if (node.kind === 'array') for (const slot of node.slots) walk(slot);
  };
  walk(map.getRoot());
  return kinds;
}

describe('HAMD basics', () => {
  it('persists', () => {
    const a = new PersistentMap<number>();
    const b = a.set('x', 1);
    expect(a.size()).toBe(0);
    expect(b.get('x')).toBe(1);
  });

  it('round-trips many keys through heterogeneous node shapes', () => {
    const map = build(Array.from({ length: 400 }, (_, i) => [`k:${i}`, i] as [string, number]));
    expect(map.size()).toBe(400);
    for (let i = 0; i < 400; i++) expect(map.get(`k:${i}`)).toBe(i);
    expect(kindsOf(map)).toContain('array');
    expect(kindsOf(map)).toContain('bitmap');
    expect(kindsOf(map)).toContain('leaf');
  });

  it('updates and deletes are persistent and structurally shared', () => {
    const a = build(Array.from({ length: 100 }, (_, i) => [`k:${i}`, i] as [string, number]));
    const b = a.set('k:42', 999);
    expect(a.get('k:42')).toBe(42);
    expect(b.get('k:42')).toBe(999);
    const c = b.delete('k:42').set('k:42', 42);
    expect(c.get('k:42')).toBe(42);
    // Setting an identical value returns the same reference.
    expect(a.set('k:42', 42)).toBe(a);
  });
});

describe('diff: identical references', () => {
  it('emits nothing and skips the whole tree when roots are the same reference', async () => {
    const map = build(Array.from({ length: 200 }, (_, i) => [`k:${i}`, i] as [string, number]));
    const result = await diffMaps(map, map);
    expect(result.events).toEqual([]);
    expect(result.stats.nodesVisited).toBe(0);
    expect(result.stats.sharedSubtreesSkipped).toBe(1);
  });

  it('emits nothing for empty maps', async () => {
    const result = await diffMaps(new PersistentMap<number>(), new PersistentMap<number>());
    expect(result.events).toEqual([]);
    expect(result.stats.nodesVisited).toBe(0);
  });
});

describe('diff: logically equal, no shared nodes', () => {
  it('finds no differences between independent copies and visits every node', async () => {
    const map = build(Array.from({ length: 300 }, (_, i) => [`k:${i}`, i] as [string, number]));
    const copy = rebuild(map);
    expect(copy.getRoot()).not.toBe(map.getRoot());
    expect(copy.nodeCount()).toBe(map.nodeCount());

    const result = await diffMaps(map, copy);
    expect(result.events).toEqual([]);
    // Nothing shared: both trees must actually be walked.
    expect(result.stats.nodesVisited).toBe(map.nodeCount() + copy.nodeCount());
    expect(result.stats.sharedSubtreesSkipped).toBe(0);
    // Comparator runs once per shared key.
    expect(result.stats.valueComparisons).toBe(300);
  });
});

describe('diff: differing node shapes', () => {
  it('compares bitmap vs array roots aligned by hash fragments', async () => {
    // Enough keys to force an ArrayNode on the second map, while the first
    // stays under the promotion threshold but shares the same key range.
    const pairs: Array<[string, number]> = [];
    // 20 keys in root fragment 0 plus one far-away anchor: bitmap root.
    const inFragment0: Array<[string, number]> = [];
    let n = 0;
    for (let i = 0; inFragment0.length < 20; i++) {
      const key = `frag0-${i}`;
      if (fragmentAt(hashKey(key), 0) === 0) inFragment0.push([key, i]);
    }
    pairs.push(...inFragment0);
    const sparse = build(pairs);
    expect(sparse.getRoot()!.kind).toBe('bitmap');

    // Add keys until the same root fragment slot set promotes to an array node.
    let densePairs = pairs.slice();
    let dense = build(densePairs);
    let guard = 0;
    while (dense.getRoot()!.kind !== 'array' && guard < 50000) {
      densePairs.push([`fill-${guard}`, guard]);
      dense = build(densePairs);
      guard++;
    }
    expect(dense.getRoot()!.kind).toBe('array');

    const result = await diffMaps(sparse, dense);
    // Every added key reported; nothing removed or changed.
    expect(result.changed).toEqual([]);
    expect(result.removed).toEqual([]);
    expect(result.added.map((e) => e.key).sort()).toEqual(
      densePairs.slice(pairs.length).map(([k]) => k).sort(),
    );
    // Shared keys were compared.
    expect(result.stats.valueComparisons).toBe(pairs.length);

    // And the reverse direction.
    const reverse = await diffMaps(dense, sparse);
    expect(reverse.added).toEqual([]);
    expect(reverse.removed.map((e) => e.key).sort()).toEqual(
      densePairs.slice(pairs.length).map(([k]) => k).sort(),
    );
  });

  it('compares leaf vs collision vs bitmap at the same slot', async () => {
    const h = hashKey('same-hash-a');
    // left: single leaf at hash h; right: a collision set (h, h, ...) plus a
    // diverging key in the same root fragment to force a bitmap wrapper.
    const left = build<number>([['same-hash-a', 1, h]]);
    expect(left.getRoot()!.kind).toBe('leaf');

    const right = build<number>([
      ['same-hash-a', 10, h],
      ['same-hash-b', 20, h],
      ['same-hash-c', 30, h],
    ]);
    expect(right.getRoot()!.kind).toBe('collision');

    const result = await diffMaps(left, right);
    expect(result.changed.map((e) => e.key)).toEqual(['same-hash-a']);
    expect(result.added.map((e) => e.key).sort()).toEqual(['same-hash-b', 'same-hash-c']);
    expect(result.removed).toEqual([]);

    // Reverse.
    const reverse = await diffMaps(right, left);
    expect(reverse.changed.map((e) => e.key)).toEqual(['same-hash-a']);
    expect(reverse.removed.map((e) => e.key).sort()).toEqual(['same-hash-b', 'same-hash-c']);
  });

  it('handles leaf vs internal subtree with an aligned fragment', async () => {
    const h = hashKey('lonely');
    const left = build<number>([['lonely', 1, h]]);
    // Right: one key sharing the root fragment but diverging deeper (bitmap),
    // and one key in a completely different root fragment.
    let alignedKey = '';
    for (let i = 0; i < 10000; i++) {
      const k = `cand-${i}`;
      const kh = hashKey(k);
      if (fragmentAt(kh, 0) === fragmentAt(h, 0) && kh !== h) {
        alignedKey = k;
        break;
      }
    }
    expect(alignedKey).not.toBe('');
    let otherKey = '';
    for (let i = 0; i < 10000; i++) {
      const k = `other-${i}`;
      if (fragmentAt(hashKey(k), 0) !== fragmentAt(h, 0)) {
        otherKey = k;
        break;
      }
    }
    const right = build<number>([
      [alignedKey, 2],
      [otherKey, 3],
    ]);
    const result = await diffMaps(left, right);
    expect(result.added.map((e) => e.key).sort()).toEqual([alignedKey, otherKey].sort());
    expect(result.removed.map((e) => e.key)).toEqual(['lonely']);
    expect(result.changed).toEqual([]);
  });
});

describe('diff: large collision sets', () => {
  it('diffs 200 same-hash keys with adds, removes and changes', async () => {
    const h = hashKey('collision-root');
    const keys = Array.from({ length: 200 }, (_, i) => `c:${i}`);
    const left = build<number>(keys.map((key) => [key, 1, h] as [string, number, number]));
    expect(left.getRoot()!.kind).toBe('collision');

    let right = left;
    // remove first 50, change next 50, keep next 50, add 50 new
    for (let i = 0; i < 50; i++) right = right.delete(`c:${i}`, h);
    for (let i = 50; i < 100; i++) right = right.set(`c:${i}`, 2, h);
    const newKeys = Array.from({ length: 50 }, (_, i) => `c:new:${i}`);
    for (const key of newKeys) right = right.set(key, 3, h);

    const result = await diffMaps(left, right);
    expect(result.removed.map((e) => e.key).sort()).toEqual(
      keys.slice(0, 50).sort(),
    );
    expect(result.changed.map((e) => e.key).sort()).toEqual(
      keys.slice(50, 100).sort(),
    );
    expect(result.added.map((e) => e.key).sort()).toEqual([...newKeys].sort());
    // Comparator only fires for the 150 keys common to both.
    expect(result.stats.valueComparisons).toBe(150);
    // Collision roots differ by reference: just the two nodes visited.
    expect(result.stats.nodesVisited).toBe(2);
  });
});

describe('diff: key removed then re-added', () => {
  it('shows nothing when a key is deleted and re-added with an equal value', async () => {
    let map = build(Array.from({ length: 100 }, (_, i) => [`k:${i}`, i] as [string, number]));
    const again = map.delete('k:31').set('k:31', 31);
    // Same logical content; set/delete normalization may share only the path,
    // but diff must report zero changes.
    const result = await diffMaps(map, again);
    expect(result.events).toEqual([]);
  });

  it('shows a change when re-added with a different value', async () => {
    let map = build<number>([['x', 1], ['y', 2], ['z', 3]]);
    const edited = map.delete('y').set('y', 42);
    const result = await diffMaps(map, edited);
    expect(result.changed.map((e) => e.key)).toEqual(['y']);
    expect(result.changed[0]).toMatchObject({ oldValue: 2, newValue: 42 });
    expect(result.added).toEqual([]);
    expect(result.removed).toEqual([]);
  });

  it('shows add+remove when deletion and re-addition happen across versions', async () => {
    const base = build<number>([['x', 1], ['y', 2]]);
    const deleted = base.delete('x');
    const readded = deleted.set('x', 9);
    const d1 = await diffMaps(base, deleted);
    expect(d1.removed.map((e) => e.key)).toEqual(['x']);
    const d2 = await diffMaps(deleted, readded);
    expect(d2.added.map((e) => e.key)).toEqual(['x']);
    const d3 = await diffMaps(base, readded);
    expect(d3.changed.map((e) => e.key)).toEqual(['x']);
  });
});

describe('diff: injected value equality', () => {
  it('uses the comparator only for keys present on both sides', async () => {
    const h = hashKey('h');
    const left = build<string>([
      ['shared-1', 'a', h],
      ['shared-2', 'b', h],
      ['only-left', 'l', h],
    ]);
    const right = build<string>([
      ['shared-1', 'A', h],
      ['shared-2', 'B', h],
      ['only-right', 'r', h],
    ]);
    const seenKeys: string[] = [];
    const result = await diffMaps(left, right, {
      valuesEqual: (a, b, key) => {
        seenKeys.push(key);
        return a.toLowerCase() === b.toLowerCase();
      },
    });
    expect(seenKeys.sort()).toEqual(['shared-1', 'shared-2']);
    // Case-insensitive equality => no changes.
    expect(result.changed).toEqual([]);
    expect(result.removed.map((e) => e.key)).toEqual(['only-left']);
    expect(result.added.map((e) => e.key)).toEqual(['only-right']);
  });

  it('reports a change when the comparator returns false', async () => {
    type Box = { id: number; rev: number };
    const left = build<Box>([['a', { id: 1, rev: 1 }]]);
    const right = build<Box>([['a', { id: 1, rev: 2 }]]);
    const byId = await diffMaps(left, right, { valuesEqual: (a, b) => a.id === b.id });
    expect(byId.changed).toEqual([]);
    const strict = await diffMaps(left, right);
    expect(strict.changed.map((e) => e.key)).toEqual(['a']);
  });
});

describe('diff: cancellation', () => {
  it('terminates immediately with an already-aborted signal', async () => {
    const map = build(Array.from({ length: 500 }, (_, i) => [`k:${i}`, i] as [string, number]));
    const copy = rebuild(map);
    const controller = new AbortController();
    controller.abort();
    const generator = diffIterate(map, copy, { signal: controller.signal });
    const result = await generator.next();
    expect(result.done).toBe(true);
    expect((result as IteratorReturnResult<{ aborted: boolean }>).value.aborted).toBe(true);

    const collected = await diffMaps(map, copy, { signal: controller.signal });
    expect(collected.stats.aborted).toBe(true);
    expect(collected.events).toEqual([]);
  });

  it('stops as soon as the signal fires during iteration', async () => {
    const map = build(Array.from({ length: 400 }, (_, i) => [`k:${i}`, i] as [string, number]));
    let edited = map;
    for (let i = 0; i < 400; i++) edited = edited.set(`k:${i}`, i + 1000);
    edited = rebuild(edited);

    const controller = new AbortController();
    const generator = diffIterate(map, edited, { signal: controller.signal });
    const events: DiffEvent<number>[] = [];
    // Abort immediately after the first event is pulled: the walk must stop at
    // the next node boundary instead of producing all 400 changes.
    const first = await generator.next();
    expect(first.done).toBe(false);
    events.push((first as IteratorYieldResult<DiffEvent<number>>).value);
    controller.abort();
    while (true) {
      const next = await generator.next();
      if (next.done) {
        expect(next.value.aborted).toBe(true);
        break;
      }
      events.push(next.value);
    }
    expect(events.length).toBeLessThan(400);
    // Events received up to cancellation stay canonically (hash-path) ordered.
    expect(events).toEqual([...events].sort(compareAsEvents));
  });
});

describe('diff: subtree sharing and stable ordering', () => {
  it('a one-key change visits almost nothing of the shared forest', async () => {
    const N = 600;
    const map = build(Array.from({ length: N }, (_, i) => [`k:${i}`, i] as [string, number]));
    const edited = map.set('k:333', -1);

    const result = await diffMaps(map, edited);
    expect(result.changed.map((e) => e.key)).toEqual(['k:333']);
    expect(result.stats.valueComparisons).toBe(1);

    // The change copies one root-path of nodes; every sibling subtree at each
    // level is identical by reference and must be skipped wholesale.
    expect(result.stats.nodesVisited).toBeLessThan(map.nodeCount() / 10);
    expect(result.stats.sharedSubtreesSkipped).toBeGreaterThan(20);
  });

  it('adding one key only flattens the one new branch', async () => {
    const N = 500;
    const map = build(Array.from({ length: N }, (_, i) => [`k:${i}`, i] as [string, number]));
    const edited = map.set('brand-new-key', 1);
    const result = await diffMaps(map, edited);
    expect(result.added.map((e) => e.key)).toEqual(['brand-new-key']);
    expect(result.removed).toEqual([]);
    expect(result.changed).toEqual([]);
    expect(result.stats.nodesVisited).toBeLessThan(map.nodeCount() / 10);
  });

  it('output is sorted by stable hash path then key', async () => {
    const left = build(Array.from({ length: 120 }, (_, i) => [`k:${i}`, i] as [string, number]));
    let right = left;
    for (let i = 0; i < 120; i++) right = i % 3 === 0 ? right.delete(`k:${i}`) : right.set(`k:${i}`, i * 7);
    right = right.set('zz-added', 1).set('aa-added', 2);

    const result = await diffMaps(left, right);
    const expected = [...result.events].sort((a, b) => {
      for (let i = 0; i < Math.max(a.path.length, b.path.length); i++) {
        const av = a.path[i] ?? -1;
        const bv = b.path[i] ?? -1;
        if (av !== bv) return av - bv;
      }
      return a.hash !== b.hash ? a.hash - b.hash : a.key.localeCompare(b.key);
    });
    expect(result.events).toEqual(expected);
    // Each bucket is sorted the same way.
    for (const bucket of [result.added, result.removed, result.changed]) {
      expect(bucket).toEqual([...bucket].sort(compareAsEvents));
    }
  });

  it('paths are derived from the key hash and are shape independent', async () => {
    const map = build(Array.from({ length: 50 }, (_, i) => [`k:${i}`, i] as [string, number]));
    const copy = rebuild(map);
    // Both equal => only changed one key to produce an event.
    const edited = copy.set('k:10', 999);
    const result = await diffMaps(map, edited);
    const event = result.changed[0];
    const expectedPath = [0, 1, 2, 3, 4, 5, 6].map((d) => fragmentAt(hashKey('k:10'), d));
    expect(event.path).toEqual(expectedPath);
    expect(event.path.length).toBe(7);
  });
});

function compareAsEvents(a: { path: number[]; hash: number; key: string }, b: { path: number[]; hash: number; key: string }): number {
  const min = Math.min(a.path.length, b.path.length);
  for (let i = 0; i < min; i++) {
    if (a.path[i] !== b.path[i]) return a.path[i] - b.path[i];
  }
  if (a.path.length !== b.path.length) return a.path.length - b.path.length;
  if (a.hash !== b.hash) return a.hash - b.hash;
  return a.key < b.key ? -1 : a.key > b.key ? 1 : 0;
}
