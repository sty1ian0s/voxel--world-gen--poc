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

### MC's approach vs our 2D strategy (2026-10-31)

This is the headline comparison the project exists for: Minecraft's 3D noise
mechanism (`mc`) against our voxel-local 2D carver (`tube`), both on the current
architecture, same seed, same band, same fresh surface-generated chunks.

Two corrections were needed to measure it at all.

**The harness measured the wrong carvers.** `benches/latency.rs` covered only
`density` and `sparse` — the original `O(N^3)` vs `O(S)` pair from `PROJECT.md`
section 1 — so `mc` and `tube` had **no measured percentiles**. Those two are
the comparison that matters and they are now the only two in the table; the
original pair keeps its own benchmark in `benches/carvers.rs`.

**The harness measured the wrong band.** It pinned `CY = 1` (world y 16..31),
which was correct when caves were placed relative to a 64-voxel box. Cave
placement is now absolute and the band is y 40..88, so the benchmark was timing
the carvers on **chunks containing no caves**. Every carver skips voxels that
are already air, so those samples measured noise and arithmetic and never a
single carve write. The band now derives from `measured_band_chunks()`, and an
assertion fails if the sampled band carves nothing.

Per-chunk carve latency, microseconds, 5000 samples each, setup excluded.
Ratios are stable to ±0.05% across runs:

| metric | mc (3D) | **tube (2D)** | **speedup** | tube + pillar | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| min | 60.12 | **7.64** | 7.9x | 9.01 | 6.7x |
| **p50** | 357.50 | **9.10** | **39.3x** | 10.50 | 34.0x |
| **p90** | 444.64 | **10.30** | **43.2x** | 11.42 | 38.9x |
| **p99** | 464.22 | **17.10** | **27.1x** | 18.02 | 25.8x |
| **p99.9** | 514.47 | **27.00** | **19.1x** | 23.83 | 21.6x |
| mean | 285.30 | **9.36** | 30.5x | 10.62 | 26.9x |
| max | 625.06 | **42.24** | 14.8x | 35.76 | 17.5x |

Tail against own median:

| carver | p90/p50 | p99/p50 | p99.9/p50 | worst chunk |
| --- | ---: | ---: | ---: | ---: |
| mc | 1.24x | 1.30x | 1.44x | 0.63 ms |
| capsule (walked 2D) | 1.15x | 1.34x | 1.66x | 0.48 ms |
| tube | 1.13x | 1.88x | 2.97x | **0.04 ms** |
| tube + pillar | 1.09x | 1.72x | 2.27x | **0.04 ms** |

**Our 2D strategy is 39x faster at the median and 19x faster at p99.9**, and the
advantage holds across the whole distribution rather than collapsing in the
tail. `tube`'s tail/median ratio is the widest of the four (2.97x at p99.9)
purely because its absolute times are so small — 9 µs — that fixed per-chunk
overheads dominate the relative spread. In absolute terms its worst chunk in
5000 is **0.04 ms against mc's 0.63 ms**, so the tail that matters for a frame
budget is still ~16x better.

**`capsule`, the walked 2D carver, is the control.** It is the approach that
preserves path memory (and can therefore express carver caves), and it lands at
**0.77x of mc's median — slower than the 3D baseline it was meant to beat**,
while `tube` is at 0.03x. That is the cost of sequential state, measured.

**Enabling the opt-in pillar type costs ~15% at p50 (9.10 → 10.50 µs)** and is
within noise at p99.9. `tube::carve` uses `default_configs()`, which disables
anything outside `CaveType::ALL`, so measuring only `tube` would have silently
excluded pillars — the first version of this table did exactly that. Both rows
are reported so the fourth type's price is visible rather than assumed.

**The sample ratio is not the latency ratio.** `tube` takes 154x fewer *noise
samples* than `mc` (27,648 vs 4,250,880) but is 39x faster in wall clock,
because `mc` skips voxels that are already air and so does far less than one
sample per voxel. Quoting 154x as a speedup would be the same category of error
as the "130x" claim corrected at the top of this file.

### The original `density` vs `sparse` percentiles

Criterion reports a median and a confidence interval; it has no notion of the
tail. World generation runs inside a frame, so the tail is what drops frames.
These are the original Pass-2 pair from `PROJECT.md` section 1, kept because the
document's headline correction above depends on them. **Not** the MC-vs-2D
comparison, which is the section above. Microseconds:

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

### Rerun after the pillar work (2026-10-31)

The per-column `powf` in the pillar gate was replaced with a precomputed
threshold (`6a34e48`), so the tables above were re-measured. **Nothing in the
recorded numbers moved for reasons attributable to that change**, and the rerun
is recorded here because it is a useful demonstration of the caveat above
rather than because it shows an effect.

Both untouched carvers came out slower than recorded, by a similar factor:

| metric | density recorded | density rerun | sparse recorded | sparse rerun |
| --- | ---: | ---: | ---: | ---: |
| min | 276.85 | 287.53 (1.04x) | 92.70 | 97.55 (1.05x) |
| p50 | 302.56 | 379.28 (1.25x) | 103.13 | 152.46 (1.48x) |
| p90 | 316.02 | 429.09 (1.36x) | 116.93 | 193.22 (1.65x) |
| p99 | 348.23 | 523.75 (1.50x) | 126.82 | 227.03 (1.79x) |
| p99.9 | 442.50 | 783.78 (1.77x) | 149.39 | 354.81 (2.38x) |

`density` and `sparse` are **not reached by the pillar change at all** -
`benches/latency.rs` imports only those two carvers - so a 1.25-1.5x shift in
both is machine state, exactly the scheduling sensitivity the caveat describes.
The ratio, which the caveat says to trust, is stable:

| | recorded | rerun |
| --- | ---: | ---: |
| sparse/density p50 | 0.34x | 0.40x |

For the carvers the change *does* touch, the shape and coverage numbers are
byte-identical to the tables below (`tube` 6.2% / 99.2% / 16 components;
`pillar` 0.5% / mean_col 14.00 / 9,216 samples), which is the result that
matters: the closed form is behaviour-preserving.

The local win was measured directly instead, since `cavestat` cannot resolve it.
On 20M values straddling the threshold, `powf` costs 5.83 ns/element against
0.58 ns for the comparison - **10x** - and an interleaved A/B of the two
revisions (7 alternating pairs, separate worktrees so nothing else varies) puts
the end-to-end `cavestat` delta at **-1.5% for pillar and +2.1% for tube with
sample SD of 9-18 ms**. The end-to-end effect is therefore *not* resolvable at
this scale, and `gen_ms` in the tables below should not be read as evidence for
or against the change.


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

### Resource usage, all five carvers (2026-10-31)

`carver_bench` is hardcoded to the original `density`/`sparse` pair, so the
resource figures for the other three were taken with `/usr/bin/time -v` around
`worldstat --side 128 --rows 8 --carver <name>`:

| carver | wall | user CPU | peak RSS |
| --- | ---: | ---: | ---: |
| **tube** | **31.25 s** | 31.24 s | 5,200 kB |
| sparse | 43.65 s | 43.64 s | 5,180 kB |
| capsule | 54.64 s | 54.63 s | 5,308 kB |
| density | 56.78 s | 56.74 s | 5,252 kB |
| mc | 143.26 s | 143.26 s | 5,216 kB |

**Peak RSS is flat across all five (5,180-5,308 kB, a 2.4% spread).** Every
carver is chunk-local with no world-sized allocation, so memory is not a
differentiator between strategies; the 128 kB spread is the residency
structures, not the carver.

**Wall time is not carver time, and this table must not be read as if it were.**
`tube` is 26x faster than `mc` at carve-only latency (13.92 us vs 366.80 us p50)
but only 4.6x faster here (31.25 s vs 143.26 s), because `worldstat` also runs
`generate_surface` per chunk and a union-find over the window to compute
connectivity and component counts. That shared work is identical for every
carver and is a large fraction of the total, so it compresses every ratio toward
1. **The `p50` table above is the per-carver number; this table is the
pipeline.**

That `tube` finishes *faster than `density`* here, while doing 2.5x the carve
cost per chunk of `sparse`, is the same conclusion the end-to-end table further
up reaches: at these settings the pipeline is dominated by surface generation
and measurement, not carving. It is why the headline comparison is per-chunk
latency rather than wall clock.

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

## Streaming: peak RSS is bounded by the window, not the region

`worldstat`, same binary, seed 1337, `--rows 16`, `/usr/bin/time -v`:

| region | chunks | voxels | peak RSS | wall |
| --- | ---: | ---: | ---: | ---: |
| 64x64 | 32,768 | 134 M | 4,816 kB | 0:44 |
| 256x256 | 524,288 | 2,147 M | 11,184 kB | 9:47 |

**16x the area for 2.3x the memory.** It is not flat, and the reason is
structural rather than noise: the sliding window carries one chunk row *per
voxel* across each boundary, so memory is `O(side)`, not `O(side²)`. Compare a
full in-RAM mask of the same region, which would exceed 1 GB.

This is the property the streaming rewrite exists for. An earlier version sized
the window per voxel instead of per chunk — multiplying the region width by 16
twice — and at 256x256 allocated a single 1.77 GB buffer, so peak RSS tracked
area exactly like the in-RAM path it was meant to replace. Measured before the
fix: 395,568 kB at 64 against 1,574,332 kB at 256.

## Cave measurement at scale

`worldstat` streams a region the in-RAM path cannot hold and reports shape
statistics per carver. Two things about its numbers must be read carefully, and
both are stated inline in its own output:

- **`chunk_connected_share` is chunk-granular and over-merges.** Components are
  linked between cave *chunks*, not cave voxels, because a chunk is the unit the
  window stores. Any two cave chunks that share a chunk merge, so this reads near
  100% where `cavestat`'s voxel-accurate figure is far lower.
- **It is NOT comparable with `cavestat`'s connectivity.** Same word, different
  question: `cavestat` judges MC-likeness on an 8x8x8 world; `worldstat`
  surveys shape at a scale no in-RAM measurement reaches.

Per-statistic scope (global vs windowed) is tabulated in the module docs of
`voxel_core::debug::window`. Summed counts and histogram quantiles are global and
exact at any window size; component count, largest component and trunk length are
windowed, and the docs state which direction each bound errs in.

### Per-carver coverage

Measured on the committed tree after absolute-Y cave placement landed. `cavestat`,
8x8x8 chunks, seed 1337, `origin_density` 2.

| carver | void% | connectivity | components | max_col | samples | cave-shaped |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `mc` (1.18 baseline) | 23.8% | 85.7% | 18,213 | 42 | 4,250,880 | no |
| `capsule` | 6.4% | 57.6% | 13 | 14 | 2,826,648 | yes |
| **`tube`** | **6.2%** | **99.2%** | **16** | **17** | **27,648** | **yes** |
| `density` | 16.3% | 38.5% | 146 | 99 | — | — |
| `sparse` | 22.4% | 3.2% | 948 | 51 | — | — |

`tube`'s cave types measured alone (the three noise types plus the opt-in
pillar):

| type | void% | connectivity | components | mean_col | max_col | samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cheese | 3.3% | 67.6% | 5 | 8.43 | 16 | 9,216 |
| spaghetti | 2.7% | 91.2% | 3 | 4.41 | 8 | 9,216 |
| noodle | 0.9% | 95.2% | 13 | 2.23 | **4** | 9,216 |
| pillar | 0.5% | 14.4% | 138 | **14.00** | 14 | 9,216 |

Pillar is **opt-in** and not part of the default world: a full-height column is
not a tunnel, so including it in the default union moved the combined carver's
`max_col` 17 → 26 and flipped its verdict from `tunnels=true` to `tunnels=false`.
`CaveType::ALL` is the three noise types; `ALL_TYPES` is the complete set. Its
`tunnels=false` in the table above is correct for a column, and its `mean_col`
of 14.00 against noodle's 2.23 is the shape the type exists to produce.

Widths order cheese > spaghetti > noodle as the wiki's mechanisms predict, and
noodle's 4-voxel column depth sits inside the wiki's 1–5 block band. Every type
passes `looks_like_tunnels`; the MC baseline does not (`max_col` 42 — its cheese
component makes genuinely tall voids, which is MC-accurate but not tunnel-shaped).

**Reading the coverage column.** The coverage figures are *not* comparable with
the historical 19–20% parity band, and the reason is measurement scope, not
carver behaviour. When cave placement became absolute, `carve_fraction*` moved
onto `measured_band_chunks()` (chunks 2..6, world y 32..95) instead of the old
chunks 1..3 (y 16..47), because the old window no longer contained the caves at
all. The taller window roughly doubles the solid rock counted while the carve
stays the same, so the ratio falls and the carved volume does not. Compare
carvers on the same band or not at all.

`mc`'s void% *rose* to 23.8% for the same reason inverted: its band widened from
23.7..46.7 to the real surface band 40..88, so it now carves where the terrain
actually is.

### What the 2D carver cannot do

**Carver caves** (main room plus 1–4 trunks and branches, 85–112 blocks long,
branching freely in 3D) are out of reach for a voxel-local carver, verified by
measurement rather than assumed. A per-column interval cannot represent a path
that revisits an `(x, z)` column at two different heights. Multiple intervals
per column — which this carver does implement — recovers *some* of that freedom
(a column can carry a cheese chamber and a noodle tunnel at once) but not
branching topology. `capsule` is the walked implementation that *can* express
paths, and it measured worse on every shape metric at 100x the samples
(57.6% connectivity, 2,826,648 samples) than the voxel-local `tube` it exists to
be compared against.

**Noise pillars** were listed here as unreachable and **are now implemented** as
an opt-in fourth type (`CaveType::Pillar`, `PILLAR_TAPER`). The earlier reasoning
— "a pillar needs neighbour dependence, which breaks `O(1)`" — was the wrong
diagnosis: a pillar needs a narrow `xz` footprint with a *full-height* column,
and that is reachable without neighbours by re-reading the same level set
through a sharpened selection gate. See the pillar row in the per-type table
above (0.5% void, `mean_col` 14.00 against noodle's 2.23).

The honest limit on pillars is **diameter**, not reachability. Gating one field
selects a thin *curve* near the level set, so measured clusters are 1–4 columns
across in `xz` — the low end of the wiki's "few blocks in diameter", closer to a
vertical vein than a chunky pillar. Getting true blobs needs the intersection of
two fields (`|a| < e1 AND |b| < e2`), which was measured to give a mean cluster
of 2.94 columns (median 2, p90 6) but costs a third field per corner — 27
samples per chunk for that type instead of 18. It is not implemented.

One measured design constraint worth knowing before retuning: connectivity
follows the `xz` footprint overlap, not column height. Raising the noodle radius
2 → 4 → 6 moved connectivity 40.4 → 40.3 → 40.1% while void doubled. The `y`
axis cannot rescue a sparse `xz` set, so `epsilon` is the coverage knob and
radius is the tunnel-height knob.

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
