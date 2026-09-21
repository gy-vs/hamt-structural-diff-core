/**
 * Persistent HAMT (hash array mapped trie) with structural-sharing-aware diffing.
 *
 * The trie branches on 5-bit hash fragments (32-way) and uses four node shapes:
 * leaves, collision nodes (full 32-bit hash collisions), bitmap-indexed nodes
 * (sparse) and array nodes (dense). All updates are persistent: `set`/`delete`
 * return a new map that shares every untouched node with the previous version.
 *
 * `map.diff(other)` compares two versions without walking shared structure:
 * when both sides reference the exact same node the whole subtree is skipped.
 * Nodes of different shapes are aligned by hash fragment, so a bitmap node can
 * be compared directly against an array node — or a leaf against a collision
 * node — while entries are still matched by their logical key. Changes are
 * emitted in a stable order (hash path from the root, key as tie-breaker) and
 * iteration is asynchronous and cancellable via `AbortSignal` or by breaking
 * out of the `for await` loop.
 */

export type Entry<V> = { key: string; value: V; hash: number };

/** Default key hash: FNV-1a, 32-bit. */
export function hashKey(value: string): number {
  let hash = 2166136261;
  for (const char of value) hash = Math.imul(hash ^ char.charCodeAt(0), 16777619);
  return hash >>> 0;
}

const BITS = 5;
const WIDTH = 1 << BITS;
const MASK = WIDTH - 1;
const MAX_SHIFT = 30;
/** Bitmap nodes convert to array nodes beyond this many children. */
const ARRAY_NODE_THRESHOLD = 16;
/** Array nodes shrink back to bitmap nodes at or below this many children. */
const ARRAY_NODE_SHRINK = 12;

export type NodeKind = 'leaf' | 'collision' | 'bitmap' | 'array';

interface LeafNode<V> {
  readonly kind: 'leaf';
  readonly hash: number;
  readonly key: string;
  readonly value: V;
}

interface CollisionEntry<V> {
  readonly key: string;
  readonly value: V;
}

interface CollisionNode<V> {
  readonly kind: 'collision';
  readonly hash: number;
  /** Kept sorted by key so traversal order is deterministic. */
  readonly entries: readonly CollisionEntry<V>[];
}

interface BitmapNode<V> {
  readonly kind: 'bitmap';
  readonly bitmap: number;
  /** Packed children: bit `i` of `bitmap` maps to `children[popcount(bitmap & ((1 << i) - 1))]`. */
  readonly children: readonly Node<V>[];
}

interface ArrayNode<V> {
  readonly kind: 'array';
  readonly count: number;
  readonly children: readonly (Node<V> | undefined)[];
}

type TerminalNode<V> = LeafNode<V> | CollisionNode<V>;
type InternalNode<V> = BitmapNode<V> | ArrayNode<V>;
type Node<V> = TerminalNode<V> | InternalNode<V>;

function isTerminal<V>(node: Node<V>): node is TerminalNode<V> {
  return node.kind === 'leaf' || node.kind === 'collision';
}

function fragment(hash: number, shift: number): number {
  return (hash >>> shift) & MASK;
}

function popcount(value: number): number {
  value -= (value >>> 1) & 0x55555555;
  value = (value & 0x33333333) + ((value >>> 2) & 0x33333333);
  value = (value + (value >>> 4)) & 0x0f0f0f0f;
  return (value * 0x01010101) >>> 24;
}

/** Child of an internal node for a 5-bit fragment index, whatever its shape. */
function childAt<V>(node: InternalNode<V>, index: number): Node<V> | undefined {
  if (node.kind === 'array') return node.children[index];
  const bit = 1 << index;
  return node.bitmap & bit ? node.children[popcount(node.bitmap & (bit - 1))] : undefined;
}

/**
 * Total order used for diff output: hash path (5-bit fragments, root-most
 * first) decides; keys break ties between entries sharing a full 32-bit hash.
 */
export function compareByHashPath(hashA: number, keyA: string, hashB: number, keyB: string): number {
  if (hashA !== hashB) {
    for (let shift = 0; shift <= MAX_SHIFT; shift += BITS) {
      const a = fragment(hashA, shift);
      const b = fragment(hashB, shift);
      if (a !== b) return a < b ? -1 : 1;
    }
  }
  return keyA < keyB ? -1 : keyA > keyB ? 1 : 0;
}

function terminalEntries<V>(node: TerminalNode<V>): readonly CollisionEntry<V>[] {
  return node.kind === 'leaf' ? [{ key: node.key, value: node.value }] : node.entries;
}

/** Combine two terminals into the smallest trie that holds both. */
function mergeTerminals<V>(
  shift: number,
  hashA: number,
  a: TerminalNode<V>,
  hashB: number,
  b: TerminalNode<V>,
): Node<V> {
  if (hashA === hashB) {
    // Identical 32-bit hashes: merge both into a single collision node.
    const entries = [...terminalEntries(a), ...terminalEntries(b)].sort((x, y) => (x.key < y.key ? -1 : 1));
    return { kind: 'collision', hash: hashA, entries };
  }
  const fragA = fragment(hashA, shift);
  const fragB = fragment(hashB, shift);
  if (fragA === fragB) {
    return { kind: 'bitmap', bitmap: 1 << fragA, children: [mergeTerminals(shift + BITS, hashA, a, hashB, b)] };
  }
  const bitmap = (1 << fragA) | (1 << fragB);
  return fragA < fragB
    ? { kind: 'bitmap', bitmap, children: [a, b] }
    : { kind: 'bitmap', bitmap, children: [b, a] };
}

function bitmapToArray<V>(bitmap: number, children: readonly Node<V>[]): ArrayNode<V> {
  const expanded: (Node<V> | undefined)[] = new Array<Node<V> | undefined>(WIDTH).fill(undefined);
  let position = 0;
  for (let index = 0; index < WIDTH; index++) {
    if (bitmap & (1 << index)) expanded[index] = children[position++];
  }
  return { kind: 'array', count: children.length, children: expanded };
}

function arrayToBitmap<V>(children: readonly (Node<V> | undefined)[]): BitmapNode<V> {
  let bitmap = 0;
  const packed: Node<V>[] = [];
  for (let index = 0; index < WIDTH; index++) {
    const child = children[index];
    if (child !== undefined) {
      bitmap |= 1 << index;
      packed.push(child);
    }
  }
  return { kind: 'bitmap', bitmap, children: packed };
}

function setNode<V>(
  node: Node<V> | undefined,
  shift: number,
  hash: number,
  key: string,
  value: V,
): { node: Node<V>; added: boolean } {
  if (node === undefined) return { node: { kind: 'leaf', hash, key, value }, added: true };
  switch (node.kind) {
    case 'leaf': {
      if (node.hash === hash && node.key === key) {
        return Object.is(node.value, value)
          ? { node, added: false }
          : { node: { kind: 'leaf', hash, key, value }, added: false };
      }
      return { node: mergeTerminals(shift, node.hash, node, hash, { kind: 'leaf', hash, key, value }), added: true };
    }
    case 'collision': {
      if (node.hash !== hash) {
        return { node: mergeTerminals(shift, node.hash, node, hash, { kind: 'leaf', hash, key, value }), added: true };
      }
      const index = node.entries.findIndex((entry) => entry.key === key);
      if (index < 0) {
        const entries = [...node.entries];
        let at = entries.length;
        while (at > 0 && entries[at - 1].key > key) at--;
        entries.splice(at, 0, { key, value });
        return { node: { kind: 'collision', hash, entries }, added: true };
      }
      if (Object.is(node.entries[index].value, value)) return { node, added: false };
      const entries = [...node.entries];
      entries[index] = { key, value };
      return { node: { kind: 'collision', hash, entries }, added: false };
    }
    case 'bitmap': {
      const index = fragment(hash, shift);
      const bit = 1 << index;
      const position = popcount(node.bitmap & (bit - 1));
      if (node.bitmap & bit) {
        const child = node.children[position];
        const result = setNode(child, shift + BITS, hash, key, value);
        if (result.node === child) return { node, added: result.added };
        const children = [...node.children];
        children[position] = result.node;
        return { node: { kind: 'bitmap', bitmap: node.bitmap, children }, added: result.added };
      }
      const children = [...node.children];
      children.splice(position, 0, { kind: 'leaf', hash, key, value });
      if (children.length > ARRAY_NODE_THRESHOLD) {
        return { node: bitmapToArray(node.bitmap | bit, children), added: true };
      }
      return { node: { kind: 'bitmap', bitmap: node.bitmap | bit, children }, added: true };
    }
    case 'array': {
      const index = fragment(hash, shift);
      const child = node.children[index];
      const result = setNode(child, shift + BITS, hash, key, value);
      if (result.node === child) return { node, added: result.added };
      const children = [...node.children];
      children[index] = result.node;
      return {
        node: { kind: 'array', count: child === undefined ? node.count + 1 : node.count, children },
        added: result.added,
      };
    }
  }
}

function deleteNode<V>(
  node: Node<V>,
  shift: number,
  hash: number,
  key: string,
): { node: Node<V> | undefined; removed: boolean } {
  switch (node.kind) {
    case 'leaf':
      return node.hash === hash && node.key === key ? { node: undefined, removed: true } : { node, removed: false };
    case 'collision': {
      if (node.hash !== hash) return { node, removed: false };
      const index = node.entries.findIndex((entry) => entry.key === key);
      if (index < 0) return { node, removed: false };
      if (node.entries.length === 2) {
        const other = node.entries[1 - index];
        return { node: { kind: 'leaf', hash, key: other.key, value: other.value }, removed: true };
      }
      const entries = [...node.entries];
      entries.splice(index, 1);
      return { node: { kind: 'collision', hash, entries }, removed: true };
    }
    case 'bitmap': {
      const index = fragment(hash, shift);
      const bit = 1 << index;
      if (!(node.bitmap & bit)) return { node, removed: false };
      const position = popcount(node.bitmap & (bit - 1));
      const child = node.children[position];
      const result = deleteNode(child, shift + BITS, hash, key);
      if (!result.removed) return { node, removed: false };
      if (result.node === undefined) {
        const bitmap = node.bitmap & ~bit;
        if (bitmap === 0) return { node: undefined, removed: true };
        const children = [...node.children];
        children.splice(position, 1);
        // A bitmap node wrapping a single terminal carries no path information: collapse.
        if (children.length === 1 && isTerminal(children[0])) return { node: children[0], removed: true };
        return { node: { kind: 'bitmap', bitmap, children }, removed: true };
      }
      const children = [...node.children];
      children[position] = result.node;
      if (children.length === 1 && isTerminal(children[0])) return { node: children[0], removed: true };
      return { node: { kind: 'bitmap', bitmap: node.bitmap, children }, removed: true };
    }
    case 'array': {
      const index = fragment(hash, shift);
      const child = node.children[index];
      if (child === undefined) return { node, removed: false };
      const result = deleteNode(child, shift + BITS, hash, key);
      if (!result.removed) return { node, removed: false };
      const count = result.node === undefined ? node.count - 1 : node.count;
      const children = [...node.children];
      children[index] = result.node;
      if (count <= ARRAY_NODE_SHRINK) {
        const packed = arrayToBitmap(children);
        if (packed.children.length === 1 && isTerminal(packed.children[0])) {
          return { node: packed.children[0], removed: true };
        }
        return { node: packed, removed: true };
      }
      return { node: { kind: 'array', count, children }, removed: true };
    }
  }
}

function lookup<V>(
  node: Node<V> | undefined,
  shift: number,
  hash: number,
  key: string,
): CollisionEntry<V> | undefined {
  let current = node;
  let level = shift;
  while (current !== undefined) {
    switch (current.kind) {
      case 'leaf':
        return current.hash === hash && current.key === key ? current : undefined;
      case 'collision':
        return current.hash === hash ? current.entries.find((entry) => entry.key === key) : undefined;
      case 'bitmap': {
        const bit = 1 << fragment(hash, level);
        if (!(current.bitmap & bit)) return undefined;
        current = current.children[popcount(current.bitmap & (bit - 1))];
        level += BITS;
        break;
      }
      case 'array': {
        const child = current.children[fragment(hash, level)];
        if (child === undefined) return undefined;
        current = child;
        level += BITS;
        break;
      }
    }
  }
  return undefined;
}

function countNodes<V>(node: Node<V> | undefined): number {
  if (node === undefined) return 0;
  switch (node.kind) {
    case 'leaf':
    case 'collision':
      return 1;
    case 'bitmap': {
      let total = 1;
      for (const child of node.children) total += countNodes(child);
      return total;
    }
    case 'array': {
      let total = 1;
      for (const child of node.children) if (child !== undefined) total += countNodes(child);
      return total;
    }
  }
}

function collectEntries<V>(node: Node<V> | undefined, out: Entry<V>[]): void {
  if (node === undefined) return;
  switch (node.kind) {
    case 'leaf':
      out.push({ key: node.key, value: node.value, hash: node.hash });
      return;
    case 'collision':
      for (const entry of node.entries) out.push({ key: entry.key, value: entry.value, hash: node.hash });
      return;
    case 'bitmap':
      for (const child of node.children) collectEntries(child, out);
      return;
    case 'array':
      for (const child of node.children) if (child !== undefined) collectEntries(child, out);
      return;
  }
}

/** A single difference between two map versions. */
export type DiffChange<V> =
  | { readonly type: 'added'; readonly key: string; readonly hash: number; readonly value: V }
  | { readonly type: 'removed'; readonly key: string; readonly hash: number; readonly value: V }
  | {
      readonly type: 'changed';
      readonly key: string;
      readonly hash: number;
      readonly oldValue: V;
      readonly newValue: V;
    };

/** Counters describing how much of the trie a diff actually touched. */
export interface DiffStats {
  /** Nodes compared or emitted. Shared subtrees (`a === b`) are never counted. */
  nodesVisited: number;
}

export interface DiffOptions<V> {
  /**
   * Value equality used to decide whether a shared key is `changed`.
   * Called only for keys present on both sides. Defaults to `Object.is`.
   */
  readonly valueEquals?: (oldValue: V, newValue: V, key: string) => boolean;
  /** Abort to stop iteration; the generator throws the signal's reason. */
  readonly signal?: AbortSignal;
  /** Optional counters, mutated in place as the diff walks the trie. */
  readonly stats?: DiffStats;
}

interface DiffContext<V> {
  readonly equals: (oldValue: V, newValue: V, key: string) => boolean;
  readonly signal?: AbortSignal;
  readonly stats?: DiffStats;
}

/** Emit every entry of a subtree (used when one side of the diff is empty). */
function* emitAll<V>(
  node: Node<V>,
  type: 'added' | 'removed',
  ctx: DiffContext<V>,
): Generator<DiffChange<V>, void, undefined> {
  ctx.signal?.throwIfAborted();
  if (ctx.stats) ctx.stats.nodesVisited++;
  switch (node.kind) {
    case 'leaf':
      yield { type, key: node.key, hash: node.hash, value: node.value };
      return;
    case 'collision':
      for (const entry of node.entries) yield { type, key: entry.key, hash: node.hash, value: entry.value };
      return;
    case 'bitmap':
      for (const child of node.children) yield* emitAll(child, type, ctx);
      return;
    case 'array':
      for (const child of node.children) if (child !== undefined) yield* emitAll(child, type, ctx);
      return;
  }
}

/** Merge two key-sorted entry lists that share the same full hash. */
function* mergeEntries<V>(
  hash: number,
  aEntries: readonly CollisionEntry<V>[],
  bEntries: readonly CollisionEntry<V>[],
  ctx: DiffContext<V>,
): Generator<DiffChange<V>, void, undefined> {
  let i = 0;
  let j = 0;
  while (i < aEntries.length && j < bEntries.length) {
    const a = aEntries[i];
    const b = bEntries[j];
    if (a.key === b.key) {
      if (!ctx.equals(a.value, b.value, a.key)) {
        yield { type: 'changed', key: a.key, hash, oldValue: a.value, newValue: b.value };
      }
      i++;
      j++;
    } else if (a.key < b.key) {
      yield { type: 'removed', key: a.key, hash, value: a.value };
      i++;
    } else {
      yield { type: 'added', key: b.key, hash, value: b.value };
      j++;
    }
  }
  while (i < aEntries.length) {
    const a = aEntries[i++];
    yield { type: 'removed', key: a.key, hash, value: a.value };
  }
  while (j < bEntries.length) {
    const b = bEntries[j++];
    yield { type: 'added', key: b.key, hash, value: b.value };
  }
}

function* diffTerminals<V>(
  a: TerminalNode<V>,
  b: TerminalNode<V>,
  ctx: DiffContext<V>,
): Generator<DiffChange<V>, void, undefined> {
  if (a.hash !== b.hash) {
    // Different logical positions: emit whichever side sorts first, then the other.
    if (compareByHashPath(a.hash, '', b.hash, '') < 0) {
      yield* emitAll(a, 'removed', ctx);
      yield* emitAll(b, 'added', ctx);
    } else {
      yield* emitAll(b, 'added', ctx);
      yield* emitAll(a, 'removed', ctx);
    }
    return;
  }
  yield* mergeEntries(a.hash, terminalEntries(a), terminalEntries(b), ctx);
}

/**
 * Compare a terminal against an internal node by walking the internal node's
 * fragment slots in order; the terminal lines up with the slot its own hash
 * fragment points to, every other child is wholly added or removed.
 */
function* diffTerminalWithInternal<V>(
  terminal: TerminalNode<V>,
  internal: InternalNode<V>,
  shift: number,
  ctx: DiffContext<V>,
  terminalIsLeft: boolean,
): Generator<DiffChange<V>, void, undefined> {
  const own = fragment(terminal.hash, shift);
  for (let index = 0; index < WIDTH; index++) {
    const child = childAt(internal, index);
    if (index === own) {
      if (terminalIsLeft) yield* diffNodes(terminal, child, shift + BITS, ctx);
      else yield* diffNodes(child, terminal, shift + BITS, ctx);
    } else if (child !== undefined) {
      yield* emitAll(child, terminalIsLeft ? 'added' : 'removed', ctx);
    }
  }
}

function* diffNodes<V>(
  a: Node<V> | undefined,
  b: Node<V> | undefined,
  shift: number,
  ctx: DiffContext<V>,
): Generator<DiffChange<V>, void, undefined> {
  if (a === b) return; // Structurally shared subtree: nothing can differ here.
  ctx.signal?.throwIfAborted();
  if (ctx.stats) ctx.stats.nodesVisited++;
  if (a === undefined || b === undefined) {
    // One side missing (they can't both be missing: a !== b): emit the other.
    if (a !== undefined) yield* emitAll(a, 'removed', ctx);
    else if (b !== undefined) yield* emitAll(b, 'added', ctx);
    return;
  }
  const aTerminal = isTerminal(a);
  const bTerminal = isTerminal(b);
  if (aTerminal && bTerminal) {
    yield* diffTerminals(a, b, ctx);
    return;
  }
  if (aTerminal) {
    yield* diffTerminalWithInternal(a, b as InternalNode<V>, shift, ctx, true);
    return;
  }
  if (bTerminal) {
    yield* diffTerminalWithInternal(b, a as InternalNode<V>, shift, ctx, false);
    return;
  }
  // Both internal: align children by hash fragment, regardless of node shape.
  for (let index = 0; index < WIDTH; index++) {
    yield* diffNodes(childAt(a, index), childAt(b, index), shift + BITS, ctx);
  }
}

export class PersistentMap<V> {
  private root: Node<V> | undefined;
  private length: number;
  private cachedNodeCount: number | undefined;

  constructor(entries: Iterable<Entry<V>> = []) {
    let root: Node<V> | undefined;
    let size = 0;
    for (const { key, value, hash } of entries) {
      const result = setNode(root, 0, hash, key, value);
      root = result.node;
      if (result.added) size++;
    }
    this.root = root;
    this.length = size;
  }

  private static fromRoot<V>(root: Node<V> | undefined, size: number): PersistentMap<V> {
    const map: PersistentMap<V> = Object.create(PersistentMap.prototype);
    map.root = root;
    map.length = size;
    map.cachedNodeCount = undefined;
    return map;
  }

  /** Number of entries in the map. */
  size(): number {
    return this.length;
  }

  /**
   * Number of trie nodes (leaves, collision, bitmap and array nodes). Compare
   * this against `DiffStats.nodesVisited` to see how much of the trie a diff
   * was able to skip through structural sharing.
   */
  nodeCount(): number {
    if (this.cachedNodeCount === undefined) this.cachedNodeCount = countNodes(this.root);
    return this.cachedNodeCount;
  }

  /** Shape of the root node, or `'empty'` for an empty map. */
  get rootKind(): NodeKind | 'empty' {
    return this.root === undefined ? 'empty' : this.root.kind;
  }

  get(key: string, hash: number = hashKey(key)): V | undefined {
    return lookup(this.root, 0, hash, key)?.value;
  }

  has(key: string, hash: number = hashKey(key)): boolean {
    return lookup(this.root, 0, hash, key) !== undefined;
  }

  set(key: string, value: V, hash: number = hashKey(key)): PersistentMap<V> {
    const result = setNode(this.root, 0, hash, key, value);
    if (result.node === this.root) return this;
    return PersistentMap.fromRoot(result.node, this.length + (result.added ? 1 : 0));
  }

  delete(key: string, hash: number = hashKey(key)): PersistentMap<V> {
    if (this.root === undefined) return this;
    const result = deleteNode(this.root, 0, hash, key);
    if (!result.removed) return this;
    return PersistentMap.fromRoot(result.node, this.length - 1);
  }

  /** All entries, ordered by hash path then key. */
  items(): Entry<V>[] {
    const out: Entry<V>[] = [];
    collectEntries(this.root, out);
    return out;
  }

  /**
   * Diff this map (old) against `other` (new). Subtrees shared by reference
   * are skipped without being walked. Changes are emitted ordered by hash
   * path, then key. Both maps must hash the same key to the same value.
   */
  async *diff(
    other: PersistentMap<V>,
    options: DiffOptions<V> = {},
  ): AsyncGenerator<DiffChange<V>, void, undefined> {
    const ctx: DiffContext<V> = {
      equals: options.valueEquals ?? Object.is,
      signal: options.signal,
      stats: options.stats,
    };
    ctx.signal?.throwIfAborted();
    for (const change of diffNodes(this.root, other.root, 0, ctx)) {
      ctx.signal?.throwIfAborted();
      yield change;
    }
  }
}

/** Standalone form of {@link PersistentMap.diff}: changes from `a` to `b`. */
export function diffMaps<V>(
  a: PersistentMap<V>,
  b: PersistentMap<V>,
  options: DiffOptions<V> = {},
): AsyncGenerator<DiffChange<V>, void, undefined> {
  return a.diff(b, options);
}
