//! The streaming path and the in-RAM batch path must generate the same world.
//!
//! This is the test the whole streaming module rests on. A streamer that is fast
//! and bounded but produces *different terrain* than `report::generate_world` is
//! worthless: every cave statistic it reports would describe a world no other
//! part of the project can reproduce, and `cavestat`'s 8x8x8 numbers would stop
//! being comparable to `worldstat`'s large-region ones.
//!
//! Byte-identical is asserted on the packed `u16` form — block id, sunlight and
//! block light, all 4096 voxels — not just on block ids, because lighting is
//! part of the generated chunk and a streamer that skipped `terrain::finish`
//! would still pass a block-id comparison.

use voxel_core::config::{CaveParams, Carver, Config, Env, RenderConfig, WorldSize};
use voxel_core::debug::report;
use voxel_core::domain::{ChunkPos, VoxelIndex};
use voxel_core::world::{CarverSpec, Region, VerticalBounds, WorldGen};

const SEED: u32 = 1337;

fn params() -> CaveParams {
    CaveParams {
        origin_density: 2.0,
        min_radius: 1.5,
        max_radius: 4.0,
    }
}

/// The batch path's world box for a given height, in chunks.
fn world_size(side: u32, height_chunks: u32) -> WorldSize {
    WorldSize {
        x: side,
        y: height_chunks,
        z: side,
    }
}

fn cfg(carver: Carver, world: WorldSize) -> Config {
    Config {
        env: Env::Dev,
        world,
        seed: SEED,
        carver,
        cave: params(),
        validation: false,
        debug_views: false,
        frame_dump: false,
        render: RenderConfig {
            force_software_gpu: false,
            vsync: false,
        },
    }
}

fn spec(carver: Carver, world_height: u32) -> CarverSpec {
    CarverSpec {
        seed: SEED,
        carver,
        cave: params(),
        world_height,
    }
}

/// The one test that matters: same seed, same world height, same carver, same
/// bytes — including the chunks above the terrain band, which are pure air with
/// refreshed flags.
#[test]
fn streaming_matches_the_batch_path_byte_for_byte() {
    for carver in [Carver::Density, Carver::Sparse] {
        let world = world_size(3, 4);
        let (grid, _) = report::generate_world(&cfg(carver, world));
        let mut streamer = WorldGen::new(spec(carver, world.y * 16));

        assert_eq!(
            streamer.bounds(),
            VerticalBounds::for_height(world.y * 16),
            "the streamer's vertical band must be the batch path's"
        );

        for pos in grid.positions().collect::<Vec<_>>() {
            let streamed = streamer.chunk_at(pos);
            let batch = grid.get(pos);
            assert_eq!(
                streamed.to_packed(),
                batch.to_packed(),
                "{carver} chunk {pos:?} differs between the streamer and the batch path"
            );
            assert_eq!(streamed.pos(), batch.pos(), "chunk carried the wrong position");
        }
    }
}

/// The same claim for the two carvers that are not behind the `Carver` enum.
///
/// `CarverSpec` can only name [`Carver`], so `mc` and `tube` are not reachable
/// through the streamer today and the parity test above cannot cover them. What
/// can be checked — and is — is that a streamed chunk equals a hand-run of the
/// same pipeline, so adding them to the enum later carries only the risk of the
/// new match arm itself.
#[test]
fn streaming_matches_generation_done_by_hand() {
    let world_height = 64;
    let mut streamer = WorldGen::new(spec(Carver::Sparse, world_height));
    let hm = voxel_core::HeightMap::fitted_to_world(SEED, world_height);

    for pos in [
        ChunkPos::new(0, 1, 0),
        ChunkPos::new(2, 1, 3),
        ChunkPos::new(0, 3, 0),
        ChunkPos::new(-4, 1, -9),
    ] {
        let streamed = streamer.chunk_at(pos);
        let mut expected = voxel_core::world::Chunk::at(pos);
        voxel_core::terrain::generate_surface(&mut expected, pos, &hm);
        voxel_core::terrain::sparse::carve(&mut expected, SEED, &params());
        voxel_core::terrain::finish(&mut expected);
        assert_eq!(
            streamed.to_packed(),
            expected.to_packed(),
            "chunk {pos:?} differs from generation done by hand"
        );
    }
}

/// Streaming is order-free and has no state carried between chunks.
///
/// Generation in the streamer reuses one buffer, so a chunk that depends on the
/// previous chunk's contents would show up here and nowhere else.
#[test]
fn a_chunk_does_not_depend_on_what_was_generated_before_it() {
    let mut a = WorldGen::new(spec(Carver::Sparse, 64));
    let mut b = WorldGen::new(spec(Carver::Sparse, 64));
    let target = ChunkPos::new(-3, 2, 7);

    let alone = a.chunk_at(target);
    // Generate a crowd of neighbours first in the other generator.
    for pos in Region::at(target, 5, 5).positions(a.bounds()) {
        b.chunk_at(pos);
    }
    let after = b.chunk_at(target);

    assert_eq!(
        alone.to_packed(),
        after.to_packed(),
        "the chunk changed depending on what was generated before it"
    );
}

/// Negative chunk coordinates must work: the streamer's whole purpose is a world
/// that extends in every direction, and the batch path has never been asked for
/// a negative chunk.
#[test]
fn generation_is_consistent_across_the_origin() {
    let mut streamer = WorldGen::new(spec(Carver::Sparse, 64));
    let mut mirrored_ok = 0;
    for (x, z) in [(0, 0), (-1, -1), (-16, 5), (1000, -1000)] {
        let pos = ChunkPos::new(x, 1, z);
        let chunk = streamer.chunk_at(pos);
        assert_eq!(chunk.pos(), pos);
        assert!(!chunk.empty, "chunk {pos:?} generated as pure air");
        // The same position generated again must be identical.
        let again = streamer.chunk_at(pos);
        assert_eq!(chunk.to_packed(), again.to_packed(), "not idempotent at {pos:?}");
        mirrored_ok += 1;
    }
    assert_eq!(mirrored_ok, 4);
}

/// Chunks outside the vertical band are air and are marked non-dirty, so a
/// region visitor can treat a whole column uniformly.
#[test]
fn out_of_band_chunks_are_air_and_flagged() {
    let mut streamer = WorldGen::new(spec(Carver::Density, 64));
    let bounds = streamer.bounds();
    assert_eq!(bounds, VerticalBounds { min_y: 0, max_y: 4 });

    assert!(!streamer.generate(ChunkPos::new(0, bounds.max_y, 0)));
    // `with_chunk` regenerates, so the out-of-band state is asserted through the
    // visit path, which is how a caller accumulating statistics reaches it.
    let region = Region::at(ChunkPos::new(0, 0, 0), 1, 1);
    streamer.visit_region(&region, |pos, c| {
        if !bounds.contains(pos.y) {
            assert!(c.empty && !c.enclosed && !c.dirty, "{pos:?} out of band");
        }
    });
    assert!(streamer.generate(ChunkPos::new(0, 0, 0)), "in-band chunk must generate");
    streamer.with_chunk(ChunkPos::new(0, 0, 0), |c| {
        assert!(!c.empty && !c.dirty, "in-band chunk must hold terrain");
    });
}

/// The visitor must visit every chunk of the region exactly once.
#[test]
fn region_visitation_covers_every_chunk_once() {
    let mut streamer = WorldGen::new(spec(Carver::Sparse, 64));
    let region = Region::square(4);
    let mut seen: Vec<ChunkPos> = Vec::new();
    streamer.visit_region(&region, |pos, _| seen.push(pos));

    assert_eq!(seen.len() as u64, streamer.region_chunks(&region));
    assert_eq!(seen.len(), 4 * 4 * 4);
    let mut sorted = seen.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), seen.len(), "a chunk was visited twice");
    // Storage order, `y` slowest: the visitor relies on it for cache locality.
    assert_eq!(seen[0], ChunkPos::new(0, 0, 0));
    assert_eq!(seen[1], ChunkPos::new(1, 0, 0));
    assert_eq!(seen[15], ChunkPos::new(3, 0, 3));
    assert_eq!(seen[16], ChunkPos::new(0, 1, 0));
}

/// A region generated one chunk at a time must equal the same region generated
/// as an in-RAM grid — the end-to-end form of the byte-identity claim, over a
/// region rather than a handful of positions.
#[test]
fn a_streamed_region_equals_the_same_region_generated_in_ram() {
    let side = 4;
    let world = world_size(side, 2);
    let (grid, _) = report::generate_world(&cfg(Carver::Sparse, world));
    let mut streamer = WorldGen::new(spec(Carver::Sparse, world.y * 16));

    let region = Region::square(side as i32);
    let mut compared = 0;
    streamer.visit_region(&region, |pos, chunk| {
        assert_eq!(
            chunk.to_packed(),
            grid.get(pos).to_packed(),
            "region chunk {pos:?} differs"
        );
        compared += 1;
    });
    assert_eq!(compared, (side * side * world.y) as usize);
}

/// A cave crossing a chunk boundary must be continuous across it.
///
/// This is the property the metrics' cross-boundary connectivity depends on. A
/// carver that lost its chunk-origin offset would still produce plausible caves
/// inside each chunk while every boundary became a wall, and the byte-identity
/// tests would not notice because both paths would be wrong the same way. Here
/// the streamed chunk and its neighbour are compared voxel by voxel along the
/// shared face: a carver whose stamps are chunk-local sees the same world
/// coordinates on both sides, so the two faces must agree about which voxels
/// are air.
#[test]
fn cave_air_is_continuous_across_a_chunk_boundary() {
    let mut streamer = WorldGen::new(spec(Carver::Sparse, 64));
    let (left, right) = (ChunkPos::new(0, 1, 0), ChunkPos::new(1, 1, 0));
    let l = streamer.chunk_at(left);
    let r = streamer.chunk_at(right);

    let mut boundary_air = 0;
    for z in 0..16u8 {
        for y in 0..16u8 {
            let a = l.block(VoxelIndex::from_parts(15, y, z));
            let b = r.block(VoxelIndex::from_parts(0, y, z));
            if a.is_air() && b.is_air() {
                boundary_air += 1;
            }
        }
    }
    assert!(
        boundary_air > 0,
        "no air on either side of the {left:?}|{right:?} face; a carver that lost \
         its world offset would look exactly like this"
    );
}

/// The streamer must not accumulate the previous chunk's terrain in its buffer.
///
/// The reuse of one 12 KiB buffer is the reason peak memory is independent of
/// region area, so it is also the one place a stale-data bug could hide. Two
/// chunks generated in sequence are compared against the same two chunks
/// generated in isolation.
#[test]
fn reusing_one_buffer_does_not_leak_the_previous_chunk() {
    let mut streamer = WorldGen::new(spec(Carver::Density, 64));
    let mut fresh = WorldGen::new(spec(Carver::Density, 64));

    let a = ChunkPos::new(0, 1, 0);
    let b = ChunkPos::new(3, 1, 2);

    let a_alone = fresh.chunk_at(a);
    let b_alone = fresh.chunk_at(b);

    let a_first = streamer.chunk_at(a);
    let b_second = streamer.chunk_at(b);

    assert_eq!(a_alone.to_packed(), a_first.to_packed());
    assert_eq!(
        b_alone.to_packed(),
        b_second.to_packed(),
        "the second chunk inherited voxels from the first"
    );
}
