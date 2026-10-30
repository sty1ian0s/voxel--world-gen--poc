# Benchmark results

Measured on this machine with `cargo bench -p voxel-core`, harness
`benches/carvers.rs`. Reproduce with:

```sh
cargo bench -p voxel-core --bench carvers -- --warm-up-time 0.5 --measurement-time 2
```

Criterion reports a confidence interval; the median is quoted below. Pass 1
surface generation is excluded from the carver timings, since both carvers pay
it and including it would damp the difference being measured.

## The headline: O(N³) versus O(S)

Carving `N x N` chunks (y fixed at 4), per-chunk work:

| chunks | density | sparse | ratio |
| ---: | ---: | ---: | ---: |
| 4 | 1.20 ms | 9.6 µs | 125x |
| 16 | 4.94 ms | 35.0 µs | 141x |
| 64 | 19.4 ms | 145 µs | 134x |
| 256 | 78.1 ms | 602 µs | 130x |

**Density scales linearly with volume:** from 4 to 256 chunks is 64x the volume
and 65.0x the time. That is the `O(N^3)` claim in `PROJECT.md` §1, confirmed
rather than assumed: the cost is one noise sample per solid voxel, so it tracks
the world's size.

**Sparse does about 130x less work per chunk**, and that ratio holds steady
across a 64x range of world sizes. It is not, however, asymptotically cheaper
here, and the reason is worth stating plainly: at the default density the cave
band spans the whole world, so cave count grows with volume too and both curves
are linear. What sparse removes is the enormous per-voxel constant, not the
growth rate. A world that grows around a *fixed* number of caves is where the
orders would visibly diverge; that case is covered by the empty-region
rejection test in the sparse module rather than by this benchmark.

## End-to-end, both carvers, same world

4x4x4 chunks = 64 chunks, `generate_world` including Pass 1:

| carver | time | throughput |
| --- | ---: | ---: |
| density | 35.8 ms | 1.79 K chunks/s |
| sparse | 24.0 ms | 2.67 K chunks/s |

Only **1.5x** apart, against the 130x difference in carver work. That is the
expected consequence of a shared Pass 1: surface generation costs ~301 µs per
chunk, about 19.3 ms for this world, and it dominates both runs. Cave carving is
simply no longer the bottleneck once the sparse carver is used, which is the
actual conclusion the POC was built to reach.

## Pipeline stages, one chunk

| stage | time |
| --- | ---: |
| setup (fill + allocate) | 0.25 µs |
| relight | 2.39 µs |
| carve_sparse | 2.39 µs |
| mesh_naive | 30.4 µs |
| mesh_greedy | 69.7 µs |
| surface | 301 µs |

Surface generation is 4x the greedy mesher and 126x the sparse carver. Lighting
and the sparse carver are both a couple of microseconds, i.e. noise beside
everything else.

## Two results that need explaining rather than hiding

**Greedy meshing is 2.3x slower than naive on CPU.** This is real and is not a
bug. On the chunk measured, greedy produces **6 quads** where naive produces
**1536** — a 256x reduction in output. Greedy pays for that with three
mask-building passes over the chunk (one per axis) plus the merge scan, whereas
naive does one early-outing sweep and emits a quad per face. Its output is
256x smaller, so it is trading CPU time for vertex buffer size, upload cost and
draw call count, which is the right trade when the GPU is the constraint and
the wrong one for a nearly-empty chunk on a fast CPU. The chain to watch is
`stages/mesh_greedy` against `stages/mesh_naive`; if the CPU side ever becomes
the bottleneck, the fix is to skip the mask build for slices that are entirely
empty or entirely full, not to abandon greedy meshing.

**Lighting costs the same for open sky and for carved terrain** (2.46 µs vs
2.42 µs). The column scan plus flood fill is cheap enough that cave complexity
is not visible at this scale, which is why the two numbers are within noise of
each other.

## Caveats

- Single machine, single run, CPU rasteriser available but no GPU benchmarked.
  Absolute times are not portable; the ratios are the point.
- `y` is fixed at 4 chunks in the scaling sweep, so the sweep measures the
  horizontal growth that `PROJECT.md` is about and not the vertical band.
- The carver benchmarks rebuild the chunk surface inside `iter_batched` setup,
  so the timed region is the carver alone and the setup cost is excluded
  deliberately rather than accidentally.
