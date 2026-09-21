/**
 * Persistent HAMT (Hash Array Mapped Trie).
 *
 * - 32-way branching, 5-bit hash fragments (7 levels for a 32-bit hash)
 * - Heterogeneous nodes: leaf / collision / bitmap / array
 * - Structural sharing: set/delete copy only the nodes on the changed path
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Entry<V> = { key: string; value: V; hash: number };

export type ValueEquality<V> = (a: V, b: V, key: string) => boolean;

export const BITS = 5;
export const BRANCHING = 1 << BITS; // 32
export const MASK = BRANCHING - 1;
export const MAX_DEPTH = Math.ceil(32 / BITS); // 7 (35 bits consumed, last fragment partial)

export type NodeKind = 'leaf' | 'collision' | 'bitmap' | 'array';

interface NodeBase {
  readonly kind: NodeKind;
  count(): number;
}

export class LeafNode<V> implements NodeBase {
  readonly kind = 'leaf' as const;
  constructor(readonly entry: Entry<V>) {}
  count(): number {
    return 1;
  }
}

export class CollisionNode<V> implements NodeBase {
  readonly kind = 'collision' as const;
  constructor(
    readonly hash: number,
    readonly entries: ReadonlyArray<Entry<V>>,
  ) {}
  count(): number {
    return this.entries.length;
  }
}

export class BitmapNode<V> implements NodeBase {
  readonly kind = 'bitmap' as const;
  constructor(
    readonly bitmap: number,
    readonly children: ReadonlyArray<HAMTNode<V>>,
  ) {}
  count(): number {
    let n = 0;
    for (const child of this.children) n += child.count();
    return n;
  }
}

export class ArrayNode<V> implements NodeBase {
  readonly kind = 'array' as const;
  // Sparse: slots are either a node or null (dense would be wasteful at low occupancy).
  readonly slots: ReadonlyArray<HAMTNode<V> | null>;
  constructor(slots: ReadonlyArray<HAMTNode<V> | null>) {
    this.slots = slots;
  }
  count(): number {
    let n = 0;
    for (const slot of this.slots) if (slot) n += slot.count();
    return n;
  }
}

export type HAMTNode<V> = LeafNode<V> | CollisionNode<V> | BitmapNode<V> | ArrayNode<V>;

// ---------------------------------------------------------------------------
// Hashing / fragments
// ---------------------------------------------------------------------------

export function hashKey(value: string): number {
  let hash = 2166136261;
  for (const char of value) hash = Math.imul(hash ^ char.charCodeAt(0), 16777619);
  return hash >>> 0;
}

/** Extract the 5-bit fragment of `hash` at trie depth `depth`. */
export function fragmentAt(hash: number, depth: number): number {
  return (hash >>> (depth * BITS)) & MASK;
}

function popcount(i: number): number {
  i = i - ((i >>> 1) & 0x55555555);
  i = (i & 0x33333333) + ((i >>> 2) & 0x33333333);
  return Math.imul((i + (i >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
}

// ---------------------------------------------------------------------------
// Trie primitives: get / assoc / remove
// ---------------------------------------------------------------------------

function getNode<V>(node: HAMTNode<V>, hash: number, key: string, depth: number): V | undefined {
  switch (node.kind) {
    case 'leaf':
      return node.entry.hash === hash && node.entry.key === key ? node.entry.value : undefined;
    case 'collision':
      if (node.hash !== hash) return undefined;
      return node.entries.find((entry) => entry.key === key)?.value;
    case 'bitmap': {
      const frag = fragmentAt(hash, depth);
      const bit = 1 << frag;
      if ((node.bitmap & bit) === 0) return undefined;
      return getNode(node.children[popcount(node.bitmap & (bit - 1))], hash, key, depth + 1);
    }
    case 'array': {
      const child = node.slots[fragmentAt(hash, depth)];
      return child ? getNode(child, hash, key, depth + 1) : undefined;
    }
  }
}

/** Two distinct entries live at the same fragment slot: descend until their hashes diverge. */
function mergeLeaves<V>(left: Entry<V>, right: Entry<V>, depth: number): HAMTNode<V> {
  if (left.hash === right.hash) {
    return new CollisionNode<V>(left.hash, [left, right]);
  }
  if (depth >= MAX_DEPTH) {
    // With a 32-bit hash all bits are consumed; only a true full-hash collision reaches here.
    return new CollisionNode<V>(left.hash, [left, right]);
  }
  const leftFrag = fragmentAt(left.hash, depth);
  const rightFrag = fragmentAt(right.hash, depth);
  if (leftFrag === rightFrag) {
    const merged = mergeLeaves(left, right, depth + 1);
    return new BitmapNode<V>(1 << leftFrag, [merged]);
  }
  if (leftFrag < rightFrag) {
    return new BitmapNode<V>((1 << leftFrag) | (1 << rightFrag), [new LeafNode(left), new LeafNode(right)]);
  }
  return new BitmapNode<V>((1 << leftFrag) | (1 << rightFrag), [new LeafNode(right), new LeafNode(left)]);
}

/** Merge a (full-hash) collision node with a leaf whose hash differs, descending from `depth`. */
function mergeCollision<V>(collision: CollisionNode<V>, leaf: LeafNode<V>, depth: number): HAMTNode<V> {
  const cFrag = fragmentAt(collision.hash, depth);
  const lFrag = fragmentAt(leaf.entry.hash, depth);
  if (cFrag === lFrag) {
    if (depth + 1 >= MAX_DEPTH) {
      // Hashes never diverge within the trie: extend the collision set with a virtual full hash.
      return new CollisionNode<V>(collision.hash, [...collision.entries, leaf.entry]);
    }
    const nested = mergeCollision(collision, leaf, depth + 1);
    return new BitmapNode<V>(1 << cFrag, [nested]);
  }
  if (cFrag < lFrag) {
    return new BitmapNode<V>((1 << cFrag) | (1 << lFrag), [collision, leaf]);
  }
  return new BitmapNode<V>((1 << cFrag) | (1 << lFrag), [leaf, collision]);
}

/** At which occupancy a BitmapNode becomes an ArrayNode (and below which it demotes back). */
const ARRAY_THRESHOLD = 16;

function assocNode<V>(node: HAMTNode<V>, entry: Entry<V>, depth: number): HAMTNode<V> {
  switch (node.kind) {
    case 'leaf': {
      if (node.entry.hash === entry.hash) {
        if (node.entry.key === entry.key) {
          if (node.entry.value === entry.value) return node;
          return new LeafNode<V>(entry);
        }
        return new CollisionNode<V>(entry.hash, [node.entry, entry]);
      }
      return mergeLeaves(node.entry, entry, depth);
    }
    case 'collision': {
      if (node.hash === entry.hash) {
        const index = node.entries.findIndex((existing) => existing.key === entry.key);
        if (index >= 0) {
          if (node.entries[index].value === entry.value) return node;
          const entries = node.entries.slice();
          entries[index] = entry;
          return new CollisionNode<V>(node.hash, entries);
        }
        return new CollisionNode<V>(node.hash, [...node.entries, entry]);
      }
      return mergeCollision(node, new LeafNode(entry), depth);
    }
    case 'bitmap': {
      const frag = fragmentAt(entry.hash, depth);
      const bit = 1 << frag;
      const index = popcount(node.bitmap & (bit - 1));
      if ((node.bitmap & bit) === 0) {
        if (popcount(node.bitmap) + 1 >= ARRAY_THRESHOLD) {
          return bitmapToArray(node.bitmap, node.children, frag, new LeafNode(entry));
        }
        const children = node.children.slice();
        children.splice(index, 0, new LeafNode(entry));
        return new BitmapNode<V>(node.bitmap | bit, children);
      }
      const child = node.children[index];
      const updated = assocNode(child, entry, depth + 1);
      if (updated === child) return node;
      const children = node.children.slice();
      children[index] = updated;
      return new BitmapNode<V>(node.bitmap, children);
    }
    case 'array': {
      const frag = fragmentAt(entry.hash, depth);
      const existing = node.slots[frag];
      if (!existing) {
        const slots = node.slots.slice();
        slots[frag] = new LeafNode(entry);
        return new ArrayNode<V>(slots);
      }
      const updated = assocNode(existing, entry, depth + 1);
      if (updated === existing) return node;
      const slots = node.slots.slice();
      slots[frag] = updated;
      return new ArrayNode<V>(slots);
    }
  }
}

/** Promote a bitmap node to an array node while inserting a child into the new fragment. */
function bitmapToArray<V>(
  bitmap: number,
  children: ReadonlyArray<HAMTNode<V>>,
  newFrag: number,
  newChild: HAMTNode<V>,
): ArrayNode<V> {
  const slots: (HAMTNode<V> | null)[] = new Array(BRANCHING).fill(null);
  let childIndex = 0;
  for (let frag = 0; frag < BRANCHING; frag++) {
    if ((bitmap & (1 << frag)) !== 0) {
      slots[frag] = children[childIndex++];
    }
  }
  slots[newFrag] = newChild;
  return new ArrayNode<V>(slots);
}

function removeNode<V>(node: HAMTNode<V>, hash: number, key: string, depth: number): HAMTNode<V> | null {
  switch (node.kind) {
    case 'leaf':
      return node.entry.hash === hash && node.entry.key === key ? null : node;
    case 'collision': {
      if (node.hash !== hash) return node;
      const index = node.entries.findIndex((entry) => entry.key === key);
      if (index < 0) return node;
      const entries = node.entries.filter((entry) => entry.key !== key);
      if (entries.length === 1) return new LeafNode<V>(entries[0]);
      return new CollisionNode<V>(node.hash, entries);
    }
    case 'bitmap': {
      const frag = fragmentAt(hash, depth);
      const bit = 1 << frag;
      if ((node.bitmap & bit) === 0) return node;
      const index = popcount(node.bitmap & (bit - 1));
      const updated = removeNode(node.children[index], hash, key, depth + 1);
      if (updated === node.children[index]) return node;
      if (updated === null) {
        const remaining = node.bitmap ^ bit;
        if (remaining === 0) return null;
        // Note: do NOT unwrap a single surviving child here — its fragments are
        // keyed at depth+1, while this node's slot is keyed at depth. Keeping the
        // BitmapNode wrapper preserves the level (matters inside ArrayNodes).
        const children = node.children.slice();
        children.splice(index, 1);
        return new BitmapNode<V>(remaining, children);
      }
      const children = node.children.slice();
      children[index] = updated;
      return new BitmapNode<V>(node.bitmap, children);
    }
    case 'array': {
      const frag = fragmentAt(hash, depth);
      const existing = node.slots[frag];
      if (!existing) return node;
      const updated = removeNode(existing, hash, key, depth + 1);
      if (updated === existing) return node;
      const slots = node.slots.slice();
      slots[frag] = updated;
      if (updated === null) {
        let occupied = 0;
        for (const slot of slots) if (slot) occupied++;
        if (occupied < ARRAY_THRESHOLD) return arrayToBitmap(slots);
      }
      return new ArrayNode<V>(slots);
    }
  }
}

function arrayToBitmap<V>(slots: ReadonlyArray<HAMTNode<V> | null>): BitmapNode<V> {
  let bitmap = 0;
  const children: HAMTNode<V>[] = [];
  for (let frag = 0; frag < BRANCHING; frag++) {
    const slot = slots[frag];
    if (!slot) continue;
    bitmap |= 1 << frag;
    children.push(slot);
  }
  return new BitmapNode<V>(bitmap, children);
}

function collectEntries<V>(node: HAMTNode<V>, out: Entry<V>[]): void {
  switch (node.kind) {
    case 'leaf':
      out.push(node.entry);
      break;
    case 'collision':
      out.push(...node.entries);
      break;
    case 'bitmap':
      for (const child of node.children) collectEntries(child, out);
      break;
    case 'array':
      for (const slot of node.slots) if (slot) collectEntries(slot, out);
      break;
  }
}

function countNodes<V>(node: HAMTNode<V> | null): number {
  if (!node) return 0;
  switch (node.kind) {
    case 'leaf':
    case 'collision':
      return 1;
    case 'bitmap': {
      let n = 1;
      for (const child of node.children) n += countNodes(child);
      return n;
    }
    case 'array': {
      let n = 1;
      for (const slot of node.slots) if (slot) n += countNodes(slot);
      return n;
    }
  }
}

// ---------------------------------------------------------------------------
// Public map
// ---------------------------------------------------------------------------

export class PersistentMap<V> {
  private readonly root: HAMTNode<V> | null;
  private readonly length: number;

  constructor(entries: ReadonlyArray<Entry<V>> = []) {
    let root: HAMTNode<V> | null = null;
    for (const entry of entries) root = root ? assocNode(root, entry, 0) : new LeafNode(entry);
    this.root = root;
    this.length = entries.length;
  }

  private static create<V>(root: HAMTNode<V> | null, length: number): PersistentMap<V> {
    const map = new PersistentMap<V>() as unknown as {
      root: HAMTNode<V> | null;
      length: number;
    };
    map.root = root;
    map.length = length;
    return map as unknown as PersistentMap<V>;
  }

  private withRoot(root: HAMTNode<V> | null, length: number): PersistentMap<V> {
    return PersistentMap.create(root, length);
  }

  get(key: string, hash: number = hashKey(key)): V | undefined {
    return this.root ? getNode(this.root, hash, key, 0) : undefined;
  }

  set(key: string, value: V, hash: number = hashKey(key)): PersistentMap<V> {
    const existing = this.get(key, hash);
    if (existing !== undefined && Object.is(existing, value)) return this;
    const entry: Entry<V> = { key, value, hash };
    const root = this.root ? assocNode(this.root, entry, 0) : new LeafNode(entry);
    return this.withRoot(root, existing !== undefined ? this.length : this.length + 1);
  }

  delete(key: string, hash: number = hashKey(key)): PersistentMap<V> {
    if (!this.root || this.get(key, hash) === undefined) return this;
    const root = removeNode(this.root, hash, key, 0);
    return this.withRoot(root, this.length - 1);
  }

  size(): number {
    return this.length;
  }

  /** Number of trie node objects (leaf / collision / bitmap / array). Public for structural tests. */
  nodeCount(): number {
    return countNodes(this.root);
  }

  getRoot(): HAMTNode<V> | null {
    return this.root;
  }

  items(): Entry<V>[] {
    const out: Entry<V>[] = [];
    if (this.root) collectEntries(this.root, out);
    return out.sort((a, b) => (a.hash !== b.hash ? a.hash - b.hash : a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
  }
}

// ---------------------------------------------------------------------------
// Structural diff
// ---------------------------------------------------------------------------

export interface DiffOptions<V> {
  /** Value equality; invoked only for keys present on both sides. Defaults to Object.is. */
  valuesEqual?: ValueEquality<V>;
  /** Aborting the signal stops the async iteration at the next node boundary. */
  signal?: AbortSignal;
}

export interface DiffEntry<V> {
  key: string;
  /** Added on the right side only / removed from the left side only. */
  type: 'added' | 'removed';
  value: V;
  /** Full 5-bit fragment path (the collision tail is the hash itself). */
  path: number[];
  hash: number;
}

export interface ChangedEntry<V> {
  key: string;
  type: 'changed';
  oldValue: V;
  newValue: V;
  path: number[];
  hash: number;
}

export type DiffEvent<V> = DiffEntry<V> | ChangedEntry<V>;

export interface DiffStats {
  /** Nodes actually descended into / inspected (each side of a pair counted separately). */
  nodesVisited: number;
  /** Subtree pairs skipped wholesale because both sides reference the identical node. */
  sharedSubtreesSkipped: number;
  /** Calls made to the injected value comparator (shared keys only). */
  valueComparisons: number;
  aborted: boolean;
}

export interface DiffResult<V> {
  added: DiffEntry<V>[];
  removed: DiffEntry<V>[];
  changed: ChangedEntry<V>[];
  /** All events in canonical (stable hash path, then key) order. */
  events: DiffEvent<V>[];
  stats: DiffStats;
}

const defaultEquality: ValueEquality<unknown> = (a, b) => Object.is(a, b);

/**
 * Walk the structural diff of two persistent HAMTs as an async iteration.
 *
 * Shared node references short-circuit an entire subtree; hash fragments align
 * logical keys even when the two sides have different node shapes
 * (bitmap vs array, leaf vs collision, ...). Yields events in canonical order
 * and cooperates with an AbortSignal.
 */
export async function* diffIterate<V>(
  before: PersistentMap<V>,
  after: PersistentMap<V>,
  options: DiffOptions<V> = {},
): AsyncGenerator<DiffEvent<V>, DiffStats> {
  const equals = (options.valuesEqual ?? defaultEquality) as ValueEquality<V>;
  const stats: DiffStats = { nodesVisited: 0, sharedSubtreesSkipped: 0, valueComparisons: 0, aborted: false };
  const aborted = (): boolean => {
    if (options.signal?.aborted) stats.aborted = true;
    return stats.aborted;
  };

  const leftRoot = before.getRoot();
  const rightRoot = after.getRoot();

  if (leftRoot === rightRoot) {
    if (leftRoot) stats.sharedSubtreesSkipped++;
  } else {
    yield* diffNodes(leftRoot, rightRoot, 0);
  }

  return stats;

  // -- terminal emitters -----------------------------------------------------

  /** Canonical full fragment path derived from the key hash: stable across node shapes. */
  function eventPath(hash: number): number[] {
    const path: number[] = [];
    for (let depth = 0; depth < MAX_DEPTH; depth++) path.push(fragmentAt(hash, depth));
    return path;
  }

  function makeEvent(type: 'added' | 'removed', entry: Entry<V>): DiffEvent<V> {
    return { key: entry.key, type, value: entry.value, path: eventPath(entry.hash), hash: entry.hash };
  }

  function makeChanged(entry: Entry<V>, other: Entry<V>): DiffEvent<V> {
    return {
      key: entry.key,
      type: 'changed',
      oldValue: entry.value,
      newValue: other.value,
      path: eventPath(entry.hash),
      hash: entry.hash,
    };
  }

  /** Compare two leaves whose fragments have already been aligned at this depth. */
  function* diffLeaves(
    left: LeafNode<V>,
    right: LeafNode<V>,
  ): Generator<DiffEvent<V>> {
    if (left.entry.hash === right.entry.hash && left.entry.key === right.entry.key) {
      stats.valueComparisons++;
      if (!equals(left.entry.value, right.entry.value, left.entry.key)) {
        yield makeChanged(left.entry, right.entry);
      }
      return;
    }
    // Different logical keys occupying the same aligned slot: one removal, one addition.
    yield* sortTwo(makeEvent('removed', left.entry), makeEvent('added', right.entry));
  }

  function* diffCollisionSets(
    left: CollisionNode<V>,
    right: CollisionNode<V>,
  ): Generator<DiffEvent<V>> {
    const rightByKey = new Map<string, Entry<V>>();
    for (const entry of right.entries) rightByKey.set(entry.key, entry);
    const seen = new Set<string>();
    const pending: DiffEvent<V>[] = [];
    for (const entry of left.entries) {
      seen.add(entry.key);
      const other = rightByKey.get(entry.key);
      if (other) {
        stats.valueComparisons++;
        if (!equals(entry.value, other.value, entry.key)) pending.push(makeChanged(entry, other));
      } else {
        pending.push(makeEvent('removed', entry));
      }
    }
    for (const entry of right.entries) {
      if (!seen.has(entry.key)) pending.push(makeEvent('added', entry));
    }
    pending.sort(compareEvents);
    yield* pending;
  }

  // -- shape-independent alignment -------------------------------------------

  /** Enumerate `(fragment, child)` pairs of an internal node in fragment order. */
  function slotsOf(node: BitmapNode<V> | ArrayNode<V>): [number, HAMTNode<V>][] {
    const out: [number, HAMTNode<V>][] = [];
    if (node.kind === 'bitmap') {
      let bitmap = node.bitmap;
      let index = 0;
      while (bitmap !== 0) {
        const frag = 31 - Math.clz32(bitmap & -bitmap);
        out.push([frag, node.children[index++]]);
        bitmap &= bitmap - 1;
      }
    } else {
      for (let frag = 0; frag < BRANCHING; frag++) {
        const slot = node.slots[frag];
        if (slot) out.push([frag, slot]);
      }
    }
    return out;
  }

  /** Yield every entry below `node` (used for wholly one-sided subtrees). */
  function* flatten(node: HAMTNode<V>): Generator<Entry<V>> {
    switch (node.kind) {
      case 'leaf':
        yield node.entry;
        break;
      case 'collision':
        for (const entry of node.entries) yield entry;
        break;
      case 'bitmap':
      case 'array':
        for (const [, child] of slotsOf(node)) yield* flatten(child);
        break;
    }
  }

  // -- core recursion --------------------------------------------------------

  function* diffNodes(
    left: HAMTNode<V> | null,
    right: HAMTNode<V> | null,
    depth: number,
  ): Generator<DiffEvent<V>> {
    if (aborted()) return;
    if (left === right) {
      // Identical reference (both non-null): the whole subtree is shared.
      if (left) stats.sharedSubtreesSkipped++;
      return;
    }
    if (!left || !right) {
      const events: DiffEvent<V>[] = [];
      if (left) for (const entry of flatten(left)) events.push(makeEvent('removed', entry));
      if (right) for (const entry of flatten(right)) events.push(makeEvent('added', entry));
      events.sort(compareEvents);
      yield* events;
      return;
    }

    stats.nodesVisited += 2;

    if (left.kind === 'bitmap' || left.kind === 'array') {
      if (right.kind === 'bitmap' || right.kind === 'array') {
        // Internal vs internal: hash-fragment alignment across heterogeneous shapes.
        const leftSlots = slotsOf(left);
        const rightSlots = slotsOf(right);
        let i = 0;
        let j = 0;
        while (i < leftSlots.length || j < rightSlots.length) {
          if (aborted()) return;
          const lf = i < leftSlots.length ? leftSlots[i][0] : BRANCHING;
          const rf = j < rightSlots.length ? rightSlots[j][0] : BRANCHING;
          if (lf === rf) {
            const [, lChild] = leftSlots[i++];
            const [, rChild] = rightSlots[j++];
            yield* diffNodes(lChild, rChild, depth + 1);
          } else if (lf < rf) {
            const [, child] = leftSlots[i++];
            yield* diffNodes(child, null, depth + 1);
          } else {
            const [, child] = rightSlots[j++];
            yield* diffNodes(null, child, depth + 1);
          }
        }
        return;
      }
      // Internal vs singleton; align via hash fragments.
      yield* diffSingletonInternal(left, right as LeafNode<V> | CollisionNode<V>, depth, true);
      return;
    }

    if (right.kind === 'bitmap' || right.kind === 'array') {
      // Singleton vs internal.
      yield* diffSingletonInternal(right, left as LeafNode<V> | CollisionNode<V>, depth, false);
      return;
    }

    // Both singletons (leaf / collision in any combination).
    if (left.kind === 'leaf' && right.kind === 'leaf') {
      yield* diffLeaves(left, right);
      return;
    }
    if (left.kind === 'collision' && right.kind === 'collision') {
      if (left.hash === right.hash) {
        yield* diffCollisionSets(left, right);
      } else {
        // Different hash sharing a prefix down to this depth: flatten both tails.
        yield* flattenDiff(left, right);
      }
      return;
    }
    // Collision vs leaf (or vice versa), same fragment at this depth.
    const collision = (left.kind === 'collision' ? left : right) as CollisionNode<V>;
    const leaf = (left.kind === 'leaf' ? left : right) as LeafNode<V>;
    const leafIsRight = right.kind === 'leaf';
    if (collision.hash === leaf.entry.hash) {
      // Same full hash: compare the key sets directly.
      const pending: DiffEvent<V>[] = [];
      const seen = new Set<string>();
      for (const entry of collision.entries) {
        seen.add(entry.key);
        if (entry.key === leaf.entry.key) {
          stats.valueComparisons++;
          if (!equals(entry.value, leaf.entry.value, entry.key)) {
            pending.push(leafIsRight ? makeChanged(entry, leaf.entry) : makeChanged(leaf.entry, entry));
          }
        } else {
          pending.push(leafIsRight ? makeEvent('removed', entry) : makeEvent('added', entry));
        }
      }
      if (!seen.has(leaf.entry.key)) {
        pending.push(leafIsRight ? makeEvent('added', leaf.entry) : makeEvent('removed', leaf.entry));
      }
      pending.sort(compareEvents);
      yield* pending;
    } else {
      yield* flattenDiff(left, right);
    }
  }

  /** Internal node on one side, singleton on the other; `singleIsRight` flips add/remove. */
  function* diffSingletonInternal(
    internal: BitmapNode<V> | ArrayNode<V>,
    single: LeafNode<V> | CollisionNode<V>,
    depth: number,
    singleIsRight: boolean,
  ): Generator<DiffEvent<V>> {
    const singleFrag =
      single.kind === 'leaf'
        ? fragmentAt(single.entry.hash, depth)
        : fragmentAt(single.hash, depth);
    let alignedChild: HAMTNode<V> | null = null;
    if (internal.kind === 'bitmap') {
      const bit = 1 << singleFrag;
      if (internal.bitmap & bit) alignedChild = internal.children[popcount(internal.bitmap & (bit - 1))];
    } else {
      alignedChild = internal.slots[singleFrag];
    }

    const events: DiffEvent<V>[] = [];
    if (alignedChild) {
      // Aligned fragment: recurse with singleton on its side, internal child on the other.
      const gen = singleIsRight
        ? diffNodes(alignedChild, single, depth + 1)
        : diffNodes(single, alignedChild, depth + 1);
      for (const event of collect(gen)) events.push(event);
    } else if (single.kind === 'leaf') {
      events.push(singleIsRight ? makeEvent('added', single.entry) : makeEvent('removed', single.entry));
    } else {
      for (const entry of single.entries) {
        events.push(singleIsRight ? makeEvent('added', entry) : makeEvent('removed', entry));
      }
    }

    // All other internal slots are wholly one-sided (the opposite side to the singleton).
    for (const [frag, child] of slotsOf(internal)) {
      if (frag === singleFrag) continue;
      for (const entry of flatten(child)) {
        events.push(singleIsRight ? makeEvent('removed', entry) : makeEvent('added', entry));
      }
    }
    events.sort(compareEvents);
    yield* events;
  }

  function* flattenDiff(left: HAMTNode<V>, right: HAMTNode<V>): Generator<DiffEvent<V>> {
    const events: DiffEvent<V>[] = [];
    for (const entry of flatten(left)) events.push(makeEvent('removed', entry));
    for (const entry of flatten(right)) events.push(makeEvent('added', entry));
    events.sort(compareEvents);
    yield* events;
  }
}

// ---------------------------------------------------------------------------
// Diff helpers
// ---------------------------------------------------------------------------

function compareEvents<V>(a: DiffEvent<V>, b: DiffEvent<V>): number {
  const min = Math.min(a.path.length, b.path.length);
  for (let i = 0; i < min; i++) {
    if (a.path[i] !== b.path[i]) return a.path[i] - b.path[i];
  }
  if (a.path.length !== b.path.length) return a.path.length - b.path.length;
  if (a.hash !== b.hash) return a.hash - b.hash;
  return a.key < b.key ? -1 : a.key > b.key ? 1 : 0;
}

function* sortTwo<V>(a: DiffEvent<V>, b: DiffEvent<V>): Generator<DiffEvent<V>> {
  if (compareEvents(a, b) <= 0) {
    yield a;
    yield b;
  } else {
    yield b;
    yield a;
  }
}

function collect<V>(gen: Generator<DiffEvent<V>>): DiffEvent<V>[] {
  const out: DiffEvent<V>[] = [];
  for (const event of gen) out.push(event);
  return out;
}

/**
 * Collect a full structural diff between two maps.
 *
 * `added` / `removed` / `changed` are each sorted by stable hash path then key.
 * `stats` exposes visited node counts so callers can verify subtree sharing.
 */
export async function diffMaps<V>(before: PersistentMap<V>, after: PersistentMap<V>, options: DiffOptions<V> = {}): Promise<DiffResult<V>> {
  const added: DiffEntry<V>[] = [];
  const removed: DiffEntry<V>[] = [];
  const changed: ChangedEntry<V>[] = [];
  const events: DiffEvent<V>[] = [];
  const generator = diffIterate(before, after, options);
  while (true) {
    const { done, value } = await generator.next();
    if (done) {
      // value carries the returned stats.
      const stats = value as DiffStats;
      events.sort(compareEvents);
      return { added, removed, changed, events, stats };
    }
    events.push(value as DiffEvent<V>);
    const event = value as DiffEvent<V>;
    if (event.type === 'added') added.push(event);
    else if (event.type === 'removed') removed.push(event);
    else changed.push(event as ChangedEntry<V>);
  }
}
