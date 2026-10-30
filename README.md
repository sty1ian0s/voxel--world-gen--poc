# voxel world-gen POC

A data-oriented voxel engine prototype in Rust that benchmarks two cave
generation paradigms on identical terrain:

1. **Full 3D density field** — continuous noise sampled per voxel, `O(N³)`.
2. **Sparse cave stamping** — deterministic origin lookup plus bounded SDF
   stamps, `O(S)`.

See [`PROJECT.md`](PROJECT.md) for the original architecture and
[`BENCHMARKS.md`](BENCHMARKS.md) for measured results.

## Status

| Stage | State |
| --- | --- |
| Config, layered dev/prod env | done, verified |
| Domain types (parse, don't validate) | done, 116 unit tests |
| Chunk storage + cull flags | done |
| Pass 1 heightmap | done |
| Pass 2A density carver | done, `O(N³)` confirmed by measurement |
| Pass 2B sparse carver | done, 130x less work per chunk |
| Debug views (ASCII, PPM/PGM, counters) | done, verified by eye |
| Greedy mesher + naive baseline | done, 256x fewer quads |
| Lighting (sunlight + block light) | done |
| Culling (frustum, flags, distance) | done |
| wgpu + winit app | **compiles; not executed — see below** |
| Criterion benchmarks | done, results committed |

### Known limitation: the render path was never run

This machine's sandbox hides `/dev/dri`, so no GPU device is reachable and
`wgpu` hangs during adapter enumeration. That is environmental, not a code
defect: a minimal standalone wgpu program with none of this project's code
hangs the same way. The renderer compiles, and its shader, vertex layout,
winding order and camera matrices are all covered by tests that need no device,
but no frame has been observed. Run this on a machine with a real GPU:

```sh
cargo run -p voxel-app --bin voxel-app            # window
cargo run -p voxel-app --bin offscreen            # one frame to a PPM
```

## Quick start

```sh
# Everything runs in dev by default; prod must be opted into.
cargo test --workspace
cargo run -p voxel-app --bin debug -- --world 4,4,4 --compare --slice 48
cargo run -p voxel-app --bin voxel-app
cargo bench -p voxel-core --bench carvers
```

## Environment

Configuration layers, lowest precedence first:

```
.env  →  .env.prod (when APP_ENV=prod)  →  .env.local  →  real environment
```

Dev is the default. Prod forces validation off so a release run cannot
accidentally benchmark with assertions enabled. Put machine-specific overrides
in `.env.local` (gitignored).

| Variable | Meaning |
| --- | --- |
| `APP_ENV` | `dev` (default) or `prod` |
| `VOXEL_WORLD_X/Y/Z` | World size in chunks |
| `VOXEL_WORLD_SEED` | Generation seed |
| `VOXEL_CARVER` | `density` (Pass 2A) or `sparse` (Pass 2B) |
| `VOXEL_CAVE_ORIGIN_DENSITY` | Cave origins per 1000 voxels³ |
| `VOXEL_CAVE_MIN/MAX_RADIUS` | Stamp radius range |
| `VOXEL_VALIDATION` | Extra debug assertions |
| `VOXEL_DEBUG_VIEWS` | Enable debug views |
| `VOXEL_FORCE_SOFTWARE_GPU` | `1` forces the CPU backend |
| `VOXEL_VSYNC` | `1` enables vsync |
| `RUST_LOG` | Log filter, e.g. `voxel_core=debug` |

## Reading the world without a GPU

The debug binary renders generation, geometry and lighting as text and images,
so correctness can be checked over SSH and in CI:

```sh
# Slice by eye, with carved air marked 'o'
cargo run -p voxel-app --bin debug -- --slice 40

# Both carvers on one world, with the work counters
cargo run -p voxel-app --bin debug -- --world 4,4,4 --compare

# Cross-sections for an image viewer
cargo run -p voxel-app --bin debug -- --slice 40 --images --out-debug debug-out
```

A correct cross-section reads as sky on top, grass (`"`), dirt (`:`), then
stone (`#`) below.

## Layout

```
crates/voxel-core    no GPU, no window: config, domain, world, terrain,
                     mesh, light, cull, debug
crates/voxel-app     wgpu + winit front end: camera, gpu, render, app,
                     plus the `debug` and `offscreen` binaries
```

## Design notes

Decisions where the code deliberately departs from a naive reading of
`PROJECT.md`, each with its reasoning in the module docs:

- **`VoxelIndex` gives `y` the highest bits** rather than following the doc's
  literal `x | (z << 4) | (y << 8)`. Same width, same bit ops, but every axis
  walk becomes unit or plane stride instead of `y` jumping 256 elements.
- **Chunk keys stay `i32` triplets**, not a packed `u64` hash. At 8 KiB per
  chunk the key size is noise next to the data it addresses, and readable
  coordinates are how placement bugs get found.
- **`ChunkPos::from_world` uses an arithmetic shift**, so world voxel -1 lands
  in chunk -1 instead of sharing chunk 0 with voxel 0.
- **Attribute-split voxel storage**, with the interleaved `u16` form confined to
  one boundary (`to_packed`) at GPU upload. Neither the mesher nor the light
  pass pays for the other's layout.
- **The naive mesher is kept** as an independent check on the greedy one. It
  caught a real bug: the naive version was letting air voxels emit faces, so
  every solid surface was reported twice.

## License

MIT — see [`LICENSE`](LICENSE).
