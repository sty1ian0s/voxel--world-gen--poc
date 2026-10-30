# Benchmark results

Comparing the two cave carvers: `density` (Pass 2A, $O(N^3)$ 3D fractal noise)
against `sparse` (Pass 2B, $O(S)$ origin lookup + SDF stamps).

Reproduce:

```sh
# per-chunk means with confidence intervals
cargo bench -p voxel-core --bench carvers -- --warm-up-time 0.5 --measurement-time 2

# per-chunk latency distribution (p50/p90/p99/p99.9)
cargo bench -p voxel-core --bench latency

# wall time, CPU time, peak RSS; wrap in perf stat for hardware counters
cargo build --release -p voxel-app --bin carver_bench
perf stat -e instructions,cycles,cache-misses,branch-misses ./target/release/carver_bench 8
```

## The headline number was wrong, and this is the correction

Earlier versions of this document reported **"sparse is 130-139x faster"**. That
number was real but it was measured at `origin_density = 0.02`, which carves
**0.2%** of the rock. At that setting both carvers operate on a nearly empty
world and the sparse carver does almost nothing, so the ratio described a world
nobody runs.

The shipped `.env` uses `origin_density = 2.0`, and that is what the parity test
asserts on. Measured there, at matched coverage:

| chunks | density | sparse | speedup |
| ---: | ---: | ---: | ---: |
| 4 | 1.187 ms | 0.416 ms | **2.85x** |
| 16 | 4.872 ms | 1.787 ms | **2.73x** |
| 64 | 19.433 ms | 7.386 ms | **2.63x** |
| 256 | 77.306 ms | 28.560 ms | **2.71x** |

**Sparse is ~2.7x faster than density, not 130x.** The old figure is retained as
a runnable benchmark (`carve/sparse_legacy_0.02`) so the inflation is
reproducible rather than merely asserted:

| chunks | density | sparse @ 0.02 | legacy speedup | sparse @ 2.0 | real speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1.187 ms | 8.0 µs | 148x | 0.416 ms | 2.85x |
| 256 | 77.306 ms | 563 µs | 137x | 28.560 ms | 2.71x |

The measurement error was a **50x overstatement** of the algorithm's advantage.
It is corrected here rather than quietly deleted.

## Why the gap is smaller than the theory suggests

`PROJECT.md` §1 frames this as $O(N^3)$ versus $O(S)$, which predicts a wide
gap. The complexity claim is **confirmed** - density's cost tracks volume
(4 -> 256 chunks is 64x the volume and 65.1x the time) - but complexity was
never the whole story at matched coverage, because the two algorithms touch a
**similar number of voxels**:

| | density | sparse |
| --- | ---: | ---: |
| voxels visited | 1,219,371 | 1,419,906 |
| carve coverage | 18.9% | 20.3% |

Sparse visits **1.16x more** voxels and still finishes 2.7x sooner. It wins on
what each visit *costs*, not on how many there are:

| | density | sparse |
| --- | ---: | ---: |
| cost per voxel visit | 228.8 ns | 170.6 ns |

A density visit is a 3D fractal noise evaluation. A sparse visit is a
squared-distance comparison against a bounded set of origins, and most visits
fail it immediately. The constant factor is where the win lives, and the
$O(S)$ framing overstates it because at realistic coverage $S$ is not small
relative to $N^3$.

## End-to-end: the carver is no longer the bottleneck

Isolating the carver flatters it, because both share Pass 1. Over a full
64-chunk world including surface generation:

| carver | end-to-end | throughput |
| --- | ---: | ---: |
| density | 34.53 ms | 1.85 K chunks/s |
| sparse | 30.03 ms | 2.13 K chunks/s |

Only **1.15x** apart, against 2.7x for the carve alone. This matches
`carver_bench`'s independent measurement (279 ms vs 242 ms over 512 chunks,
1.15x) exactly. Pass 1 costs ~301 µs/chunk and dominates both runs, so once the
sparse carver is used, **cave carving stops being what world generation spends
its time on**. That is the actual conclusion the POC was built to reach, and it
survives the correction - arguably it is strengthened, since the carve is now
known to be an even smaller share of the total than the old numbers implied.

## Latency distribution (p50/p90/p99/p99.9)

Criterion reports a median and a confidence interval; it has no notion of the
tail. World generation runs inside a frame, so the tail is what drops frames.
`benches/latency.rs` measures 5000 single-chunk carves per carver, setup
excluded, chunk positions cycled so no lucky chunk repeats. Microseconds:

| metric | density | sparse | sparse/density |
| --- | ---: | ---: | ---: |
| min | 276.85 | 92.70 | 0.33x |
| p50 | 302.56 | 103.13 | 0.34x |
| p90 | 316.02 | 116.93 | 0.37x |
| p99 | 348.23 | 126.82 | 0.36x |
| p99.9 | 442.50 | 149.39 | 0.34x |
| mean | 302.31 | 104.56 | 0.35x |
| max | 1823.22 | 194.34 | 0.11x |

Sparse is **~3x faster at every percentile**, and the ratio holds into the tail
rather than degrading. As a fraction of its own median:

| | p90/p50 | p99/p50 | p99.9/p50 |
| --- | ---: | ---: | ---: |
| density | 1.04x | 1.15x | 1.46x |
| sparse | 1.13x | 1.23x | 1.45x |

The prediction that sparse would be *spikier* - because its cost depends on how
many origins happen to land near a chunk - **did not hold**. Relative tail
spread is nearly identical, and in absolute terms sparse's worst chunk (194 µs)
is far below density's (1823 µs). The origin count per region is evidently
tight enough, at density 2.0, that the Poisson variance expected from the
design does not show up above the noise floor. Density's occasional 1.8 ms
outlier is the interesting tail here, not sparse's.

**Caveat:** absolute tail microseconds are scheduling-sensitive. Across repeated
runs of an identical binary, medians moved ~30% and p99.9 moved by up to 3x.
Within-run percentile *shape* is trustworthy; the exact tail figure is not, and
neither is the max column. Treat the ratios as the result.

## Resource usage

`carver_bench 8` (512 chunks, sequential phases in one process):

| carver | wall | user CPU | peak RSS | carved |
| --- | ---: | ---: | ---: | ---: |
| density | 279.0 ms | 270 ms | 8.57 MB | 164,766 |
| sparse | 242.2 ms | 240 ms | 8.57 MB | 220,414 |

Peak RSS is **identical (8.57 MB)** - both carvers allocate the same chunk grid,
and neither allocates meaningfully per-carve. Memory is not a differentiator;
the sparse carver's origin enumeration reuses stack and a bounded region scan
rather than building a per-chunk list.

Hardware counters, three consecutive runs at side 8 (stable to ±0.3%):

| counter | value |
| --- | ---: |
| instructions | 11.891 G |
| cycles | 3.338 G |
| IPC | 3.56 |
| branches | 1.058 G |
| branch misses | 6.879 M |
| branch misprediction rate | **0.65%** of all branches |
| cache misses | ~163 K |

An IPC of 3.56 on a 16-thread box, with only ~163 K cache misses across 11.9 G
instructions and a 0.65% misprediction rate, says this is **compute-bound in a
tight, cache-resident loop**, not memory-bound and not branch-bound. That is
consistent with both algorithms working on a single 16³ chunk (64 KB) that fits
comfortably in L2.

## Downstream effects: what carving changes for culling and meshing

The carvers do not only differ in speed. They leave the world in different
states, and those states are what the *next* stage pays for. Measured over the
same world at three sizes (`carver_bench`'s per-chunk flags):

| world | carver | empty chunks | **enclosed chunks** |
| --- | --- | ---: | ---: |
| 4x4x4 | density | 16 | 0 |
| 4x4x4 | sparse | 16 | 0 |
| 8x8x8 | density | 185 | **10** |
| 8x8x8 | sparse | 181 | **0** |
| 12x12x12 | density | 629 | **38** |
| 12x12x12 | sparse | 629 | **0** |

`empty` (all air) and `enclosed` (all solid) are the two cheap culls
`cull::CullStats` performs with a single bool read before any geometry work, so
an enclosed chunk is a chunk the mesher and renderer skip entirely.

**Sparse leaves zero enclosed chunks at every size.** Density leaves 38 at
12x12x12. This is a real, measurable rendering cost of the sparse approach: it
is faster to carve and slightly more expensive downstream. It is not a bug.
Noise-threshold carving leaves deep rock pockets untouched and fully solid,
whereas a stamp field spreads evenly enough to breach nearly every deep chunk.
The flag is set correctly in both cases; the worlds genuinely differ.

The practical reading: sparse trades **~2.7x carve time** against **a handful of
extra chunks that must be meshed and drawn**. At 38 chunks out of 1728 (2.2%)
that is a small downstream cost against a large upstream win, but it is a real
one and it grows with world size. A hybrid - sparse carve plus a cheap
enclosure sweep that re-solidifies surviving pockets - would recover both, and
is the obvious next step if the downstream cost ever matters.

The mesher side of the same question is already measured: greedy meshing emits
**6 quads** where naive emits **1536** on a typical chunk, a 256x reduction.
That difference dwarfs the 2.2% chunk-count effect here, which is why the
enclosed-chunk gap is worth recording but not worth optimising yet.

## Wattage: not measured

**No energy figure is reported, because none was measured.** This machine
exposes no `/sys/class/powercap`, no RAPL, and no `cpufreq` to this shell, and
`perf` energy events are unavailable, so a real watt reading was impossible.

What can be said without inventing a number: energy is time x power, and the
cycles column above is the physically meaningful proxy. Sparse does the same
job in **93% of the wall time and 93% of the CPU time** of density, so at equal
power draw it uses roughly **7% less energy per world**. That is an inference
from measured time, not a measurement of power, and it assumes both carvers
sustain comparable power - which is plausible for two compute-bound loops on
the same data but is an assumption.

To make it a real measurement you need a host whose kernel exposes a power PMU.
This one does not: `perf list` shows no `power`/`energy`/`rapl` event at all, so
the obvious command fails outright rather than reporting zero:

```sh
# On this box (WSL2, kernel 6.18.54.1-microsoft-standard-WSL2):
$ perf stat -e power/energy-pkg/ ./target/release/carver_bench 8
event syntax error: 'power/energy-pkg/'
                     \___ Bad event or PMU
Unable to find PMU or event on a PMU of 'power'
```

On bare-metal Linux with `CONFIG_INTEL_RAPL` (or the AMD equivalent) and
`/sys/class/powercap` present, that same command works and `perf stat -e
power/energy-pkg/` becomes a real reading. Check first with `perf list | grep -i
energy`; if that is empty, no command will produce a watt figure.

## Pipeline stages, one chunk

| stage | time |
| --- | ---: |
| setup (fill + allocate) | 0.24 µs |
| relight | 3.26 µs |
| carve_sparse | 122.5 µs |
| mesh_naive | 32.0 µs |
| mesh_greedy | 74.9 µs |
| surface | 301 µs |

`carve_sparse` now reads 122.5 µs where this document previously said 2.39 µs.
That is the 0.02 -> 2.0 density change, not a regression: the old figure was the
near-empty world. Most of that 122.5 µs is the region enumeration, which is a
fixed 27-region scan per chunk regardless of how many origins are found.

Relative to surface generation the ordering is unchanged - surface is still
2.5x the sparse carver and 4x the greedy mesher.

## Two results that need explaining rather than hiding

**Greedy meshing is 2.3x slower than naive on CPU.** Real, and not a bug. On the
chunk measured, greedy produces **6 quads** where naive produces **1536** - a
256x reduction in output. Greedy pays with three mask-building passes plus the
merge scan, and buys a 256x smaller vertex buffer, less upload traffic and fewer
draw calls. That is the right trade when the GPU is the constraint and the wrong
one for a nearly-empty chunk on a fast CPU. If the CPU side ever becomes the
bottleneck the fix is to skip the mask build for slices that are entirely empty
or entirely full, not to abandon greedy meshing.

**Lighting costs the same for open sky and for carved terrain** (2.32 µs vs
3.28 µs). The column scan plus flood fill is cheap enough that cave complexity
barely registers at this scale, which is why the two are within a microsecond.

## Caveats

- Single machine (WSL2, 16 threads), single run unless stated. Absolute times
  are not portable; the ratios are the point.
- No GPU was benchmarked. CPU rasterisation only.
- The latency tail is scheduling-sensitive; see above.
- `y` is fixed at 4 chunks in the criterion scaling sweep, so the sweep measures
  the horizontal growth `PROJECT.md` is about, not the vertical band.
- The carver benchmarks rebuild the chunk surface inside `iter_batched` setup,
  so the timed region is the carver alone and setup is excluded deliberately.
- Wattage is inferred from CPU time, not measured. Labelled as such above.
