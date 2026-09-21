# Persistent HAMT core

TypeScript library for immutable hash maps. The HAMT branches on 5-bit hash
fragments and uses leaf, collision, bitmap-indexed and array nodes; every
`set`/`delete` returns a new `PersistentMap` that shares untouched nodes with
the previous version.

## Diff API

`map.diff(other, options)` (or `diffMaps(a, b, options)`) asynchronously
iterates the `added` / `removed` / `changed` entries between two versions:

- Subtrees shared by reference are skipped without being walked, so a small
  change costs O(depth), not O(size). `map.nodeCount()` exposes the trie size
  and `options.stats.nodesVisited` reports how many nodes a diff touched.
- Nodes of different shapes are aligned by hash fragment, so heterogeneous
  node combinations (leaf/collision/bitmap/array) compare correctly.
- `options.valueEquals(oldValue, newValue, key)` customizes value comparison
  and is only called for keys present on both sides.
- Output is ordered by hash path (root-most fragment first) with the key as
  tie-breaker; `compareByHashPath` is exported for consumers.
- Iteration is cancellable via `options.signal` (AbortSignal) or by breaking
  out of the `for await` loop.

Run `npm install`, then `npm test` and `npm run build`.
