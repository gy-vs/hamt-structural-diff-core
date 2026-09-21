import { expect, it } from 'vitest';
import {
  PersistentMap,
  diffMaps,
  type DiffEvent,
  type HAMTNode,
  BRANCHING,
  fragmentAt,
} from '../../src/index.js';

/** Verify every node is internally consistent (bitmap popcount == child count, etc.). */
function assertWellFormed<V>(node: HAMTNode<V> | null, hashOf: (e: { key: string }) => number, depth = 0): void {
  if (!node) return;
  switch (node.kind) {
    case 'leaf':
      expect(fragmentAt(node.entry.hash, depth)).toBeDefined();
      break;
    case 'collision':
      expect(node.entries.length).toBeGreaterThanOrEqual(2);
      for (const e of node.entries) expect(e.hash).toBe(node.hash);
      break;
    case 'bitmap': {
      let pop = 0;
      for (let b = node.bitmap; b; b &= b - 1) pop++;
      expect(node.children.length).toBe(pop);
      for (const child of node.children) {
        // Child fragments must belong to the next level.
        assertWellFormed(child, hashOf, depth + 1);
      }
      break;
    }
    case 'array': {
      let occupied = 0;
      for (const slot of node.slots) {
        if (slot) {
          occupied++;
          assertWellFormed(slot, hashOf, depth + 1);
        }
      }
      expect(node.slots.length).toBe(BRANCHING);
      // array nodes must never sit below demotion threshold
      expect(occupied).toBeGreaterThanOrEqual(16);
      break;
    }
  }
}

it('fuzz: trie content, structural invariants, and diff against a reference map', async () => {
  let rng = 987654321;
  const rand = () => {
    rng ^= rng << 13;
    rng ^= rng >>> 17;
    rng ^= rng << 5;
    rng >>>= 0;
    return rng / 0xffffffff;
  };
  const alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const keyOf = () =>
    Array.from({ length: 1 + Math.floor(rand() * 4)}, () => alphabet[Math.floor(rand() * alphabet.length)]).join('');
  const hashOf = () => Math.floor(rand() * 0xffffffff + 1) >>> 0;

  for (let round = 0; round < 60; round++) {
    const hashes = new Map<string, number>();
    const H = (k: string) => {
      if (!hashes.has(k)) {
        // Occasionally force a shared hash to exercise collision nodes heavily.
        hashes.set(k, round % 4 === 0 && rand() < 0.25 ? 42 : hashOf());
      }
      return hashes.get(k)!;
    };

    let ref = new Map<string, number>();
    let hamt = new PersistentMap<number>();
    for (let step = 0; step < 500; step++) {
      const key = keyOf();
      const h = H(key);
      const r = rand();
      if (r < 0.6) {
        const v = Math.floor(rand() * 1000);
        ref.set(key, v);
        hamt = hamt.set(key, v, h);
      } else {
        ref.delete(key);
        hamt = hamt.delete(key, h);
      }
      if (step % 50 === 0) assertWellFormed(hamt.getRoot(), (e) => H(e.key));
    }
    assertWellFormed(hamt.getRoot(), (e) => H(e.key));
    expect(hamt.size()).toBe(ref.size);
    for (const [k, v] of ref) expect(hamt.get(k, H(k))).toBe(v);

    // Version 2 from random edits.
    let ref2 = new Map(ref);
    let hamt2 = hamt;
    for (let step = 0; step < 200; step++) {
      const key = keyOf();
      const h = H(key);
      if (rand() < 0.6) {
        const v = Math.floor(rand() * 1000);
        ref2.set(key, v);
        hamt2 = hamt2.set(key, v, h);
      } else {
        ref2.delete(key);
        hamt2 = hamt2.delete(key, h);
      }
    }
    assertWellFormed(hamt2.getRoot(), (e) => H(e.key));
    for (const [k, v] of ref2) expect(hamt2.get(k, H(k))).toBe(v);

    const result = await diffMaps(hamt, hamt2);
    const expAdded = [...ref2].filter(([k]) => !ref.has(k)).map(([k]) => k).sort();
    const expRemoved = [...ref].filter(([k]) => !ref2.has(k)).map(([k]) => k).sort();
    const expChanged = [...ref].filter(([k]) => ref2.has(k) && ref.get(k) !== ref2.get(k)).map(([k]) => k).sort();
    expect(result.added.map((e) => e.key).sort()).toEqual(expAdded);
    expect(result.removed.map((e) => e.key).sort()).toEqual(expRemoved);
    expect(result.changed.map((e) => e.key).sort()).toEqual(expChanged);
    for (const e of result.added) expect(e.value).toBe(ref2.get(e.key));
    for (const e of result.removed) expect(e.value).toBe(ref.get(e.key));

    // Independent rebuild: same logical content => empty diff, full visit.
    let rebuilt = new PersistentMap<number>();
    for (const [k, v] of ref2) rebuilt = rebuilt.set(k, v, H(k));
    const same = await diffMaps(hamt2, rebuilt);
    expect(same.events).toEqual([]);
    expect(same.stats.sharedSubtreesSkipped).toBe(0);

    const cmp = (a: DiffEvent<number>, b: DiffEvent<number>) => {
      for (let i = 0; i < Math.max(a.path.length, b.path.length); i++) {
        const av = a.path[i] ?? -1;
        const bv = b.path[i] ?? -1;
        if (av !== bv) return av - bv;
      }
      return a.hash !== b.hash ? a.hash - b.hash : a.key < b.key ? -1 : a.key > b.key ? 1 : 0;
    };
    expect(result.events).toEqual([...result.events].sort(cmp));
  }
});
