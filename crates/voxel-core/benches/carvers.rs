//! Carver benchmarks: `O(N^3)` density versus `O(S)` sparse stamping.
//!
//! This is the measurement the project exists to make (`PROJECT.md` §1).
//! Criterion is used rather than a bare `Instant` because it reports an
//! interval rather than a point estimate, and because the setup cost here is
//! large enough relative to the sparse carver that a single-shot timing would
//! mostly measure the setup.
//!
//! Pass 1 surface generation is excluded from the carver timings: both carvers
//! pay it, so including it would damp the very difference being measured.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use voxel_core::config::CaveParams;
use voxel_core::domain::{BlockId, ChunkPos};
use voxel_core::terrain::{self, density, heightmap::HeightMap, sparse};
use voxel_core::world::Chunk;
use voxel_core::{Carver, Config, Env, RenderConfig, WorldSize};

/// World sizes in chunks along x and z. `y` is fixed at the terrain band so the
/// surface pass does not dominate at any size.
const SIZES: [i32; 4] = [2, 4, 8, 16];

/// The density the shipped `.env` actually uses, and the one the parity test
/// asserts on. A previous version of this bench used 0.02, which carved 0.2% of
/// the rock; both carvers do far less work on a nearly-empty world, so the
/// speedup that setting produced described a world nobody runs.
const SPARSE_DENSITY: f32 = 2.0;

/// The setting the original 130x figure was measured at, kept benchmarkable so
/// the inflation is reproducible rather than merely asserted in the docs.
const LEGACY_SPARSE_DENSITY: f32 = 0.02;

fn cave_params(origin_density: f32) -> CaveParams {
    CaveParams {
        origin_density,
        min_radius: 1.5,
        max_radius: 4.0,
    }
}

/// Chunks with surfaces already generated, at the level the cave band is in.
fn prepared(side: i32, seed: u32) -> Vec<Chunk> {
    let hm = HeightMap::fitted_to_world(seed, 64);
    let mut out = Vec::with_capacity((side * side) as usize);
    for z in 0..side {
        for x in 0..side {
            let pos = ChunkPos::new(x, 1, z);
            let mut c = Chunk::at(pos);
            terrain::generate_surface(&mut c, pos, &hm);
            out.push(c);
        }
    }
    out
}

fn bench_carvers(c: &mut Criterion) {
    let seed = 1337u32;
    let params = cave_params(SPARSE_DENSITY);

    let mut group = c.benchmark_group("carve");
    // Re-running a carver over an already-carved chunk does strictly less work
    // because already-air voxels are skipped, so each sample restores the
    // surface first. `iter_batched` makes that explicit instead of letting the
    // setup cost silently disappear from the measurement.
    group.sample_size(20);

    for side in SIZES {
        let chunks = (side * side) as u64;
        group.throughput(Throughput::Elements(chunks));

        group.bench_with_input(BenchmarkId::new("density", chunks), &side, |b, &side| {
            b.iter_batched(
                || prepared(side, seed),
                |mut chunks| {
                    for chunk in &mut chunks {
                        black_box(density::carve(chunk, seed));
                    }
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("sparse", chunks), &side, |b, &side| {
            b.iter_batched(
                || prepared(side, seed),
                |mut chunks| {
                    for chunk in &mut chunks {
                        black_box(sparse::carve(chunk, seed, &params));
                    }
                },
                BatchSize::LargeInput,
            );
        });

        // The old near-empty setting, kept so the doc's correction is backed by
        // a runnable benchmark. Expect this to look far better than "sparse"
        // above; that gap is the point.
        let legacy = cave_params(LEGACY_SPARSE_DENSITY);
        group.bench_with_input(
            BenchmarkId::new("sparse_legacy_0.02", chunks),
            &side,
            |b, &side| {
                b.iter_batched(
                    || prepared(side, seed),
                    |mut chunks| {
                        for chunk in &mut chunks {
                            black_box(sparse::carve(chunk, seed, &legacy));
                        }
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

/// Both carvers over a whole world, so the end-to-end difference is visible
/// next to the per-chunk one.
fn bench_carver_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("carve_world");
    group.sample_size(10);

    for carver in [Carver::Density, Carver::Sparse] {
        let cfg = Config {
            env: Env::Dev,
            world: WorldSize { x: 4, y: 4, z: 4 },
            seed: 1337,
            carver,
            cave: cave_params(SPARSE_DENSITY),
            validation: false,
            debug_views: false,
            frame_dump: false,
            render: RenderConfig {
                force_software_gpu: false,
                vsync: false,
            },
        };
        group.throughput(Throughput::Elements(cfg.world.chunk_count()));
        group.bench_function(carver.to_string(), |b| {
            b.iter(|| black_box(voxel_core::debug::report::generate_world(&cfg)));
        });
    }

    group.finish();
}

/// Each pipeline stage in isolation, so a change can be attributed to a stage
/// rather than to "generation got slower".
fn bench_pipeline_stages(c: &mut Criterion) {
    let seed = 1337u32;
    let params = cave_params(SPARSE_DENSITY);
    let hm = HeightMap::fitted_to_world(seed, 64);
    let pos = ChunkPos::new(0, 1, 0);

    // A carved chunk, reused as the input for the meshing and lighting stages,
    // so those measure their own work rather than the carver's.
    let carved = {
        let mut c = Chunk::at(pos);
        terrain::generate_surface(&mut c, pos, &hm);
        sparse::carve(&mut c, seed, &params);
        c
    };

    let mut group = c.benchmark_group("stages");

    group.bench_function("surface", |b| {
        b.iter(|| {
            let mut c = Chunk::at(pos);
            terrain::generate_surface(&mut c, pos, &hm);
            black_box(c)
        });
    });

    group.bench_function("carve_sparse", |b| {
        b.iter_batched(
            || {
                let mut c = Chunk::at(pos);
                terrain::generate_surface(&mut c, pos, &hm);
                c
            },
            |mut c| black_box(sparse::carve(&mut c, seed, &params)),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("mesh_greedy", |b| {
        b.iter(|| black_box(voxel_core::mesh::mesh_chunk(&carved)));
    });

    group.bench_function("mesh_naive", |b| {
        b.iter(|| black_box(voxel_core::mesh::mesh_chunk_naive(&carved)));
    });

    group.bench_function("relight", |b| {
        b.iter_batched(
            || carved.clone(),
            |mut c| {
                voxel_core::light::relight_chunk(&mut c);
                black_box(c)
            },
            BatchSize::SmallInput,
        );
    });

    // The cost of just resetting a chunk, which is the floor for any batched
    // measurement above and makes the setup overhead visible.
    group.bench_function("setup_clear", |b| {
        b.iter(|| {
            let mut c = Chunk::at(pos);
            c.fill(BlockId::STONE);
            black_box(c)
        });
    });

    group.finish();
}

/// Lighting alone, since it is the only stage that walks a queue rather than
/// a regular grid, and so scales differently from the rest.
fn bench_lighting(c: &mut Criterion) {
    let seed = 1337u32;
    let params = cave_params(SPARSE_DENSITY);
    let hm = HeightMap::fitted_to_world(seed, 64);
    let pos = ChunkPos::new(0, 1, 0);

    let mut group = c.benchmark_group("light");

    // Open sky and a carved chunk light very differently: the first is a column
    // scan with nothing to spill, the second is dominated by the flood fill
    // escaping the caves.
    for (name, carve) in [("open_sky", false), ("carved", true)] {
        group.bench_function(name, |b| {
            b.iter_batched(
                || {
                    let mut c = Chunk::at(pos);
                    terrain::generate_surface(&mut c, pos, &hm);
                    if carve {
                        sparse::carve(&mut c, seed, &params);
                    }
                    c
                },
                |mut c| {
                    voxel_core::light::relight_chunk(&mut c);
                    black_box(c)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_carvers,
    bench_carver_comparison,
    bench_pipeline_stages,
    bench_lighting
);
criterion_main!(benches);
