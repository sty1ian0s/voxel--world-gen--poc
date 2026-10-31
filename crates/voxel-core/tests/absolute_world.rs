//! The world is one infinite, seed-driven world; a world height is a window.
//!
//! These two properties are the whole point of the absolute-Y band, and they are
//! the reason `HeightMap::fitted_to_world` stopped taking the world height
//! seriously:
//!
//! 1. **`(seed, x, z) -> surface height` does not mention the world size.** Two
//!    runs of the same seed at different heights must put the surface at the same
//!    absolute Y.
//! 2. **Overlapping windows agree byte-for-byte on their overlap.** A "world
//!    height" clips which chunks are generated; it does not change them.
//!
//! Property 2 is the real acceptance criterion: property 1 is about one number,
//! property 2 is about every voxel of the terrain, both light channels and the
//! cave air included.

use voxel_core::config::{CaveParams, Carver};
use voxel_core::domain::{ChunkPos, VoxelIndex};
use voxel_core::world::{CarverSpec, WorldGen};

const SEED: u32 = 1337;

fn params() -> CaveParams {
    CaveParams {
        origin_density: 2.0,
        min_radius: 1.5,
        max_radius: 4.0,
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

/// Property 1, at the level of a single number: the surface height must be
/// identical for every window, including windows that do not contain it.
#[test]
fn surface_height_ignores_world_size() {
    use voxel_core::HeightMap;

    let small = HeightMap::fitted_to_world(SEED, 64);
    let large = HeightMap::fitted_to_world(SEED, 512);
    let tiny = HeightMap::fitted_to_world(SEED, 16);

    let mut outside_some = 0;
    for x in -64..64 {
        for z in [-33, -1, 0, 1, 47] {
            let h = small.height(x, z);
            assert_eq!(h, large.height(x, z), "height moved with the window at ({x},{z})");
            assert_eq!(h, tiny.height(x, z), "height moved with the window at ({x},{z})");
            assert_eq!(h, HeightMap::absolute(SEED).height(x, z), "absolute differs");
            if h >= 64 {
                outside_some += 1;
            }
        }
    }
    assert!(
        outside_some > 0,
        "the sample must include columns above the small window or this proves nothing"
    );
}

/// Sea level must be absolute too, or a short window has no ocean *and* no sky
/// above the surface, and `Chunk::enclosed` can never be set — which silently
/// costs the renderer its solid-rock culling.
#[test]
fn sea_level_is_absolute_not_derived_from_the_window() {
    use voxel_core::HeightMap;

    for height in [16u32, 32, 64, 128, 512] {
        let hm = HeightMap::fitted_to_world(SEED, height);
        assert_eq!(
            hm.sea_level(),
            voxel_core::terrain::heightmap::SEA_LEVEL,
            "sea level tracked the window at height {height}"
        );
    }

    // The degenerate case the old box-derived sea level created: a window whose
    // sea level sat at or above its own ceiling, leaving no column that could
    // reach the top of a chunk. Sea level must now be an ordinary absolute Y.
    let tall = 512u32;
    let short = 64u32;
    assert_eq!(
        HeightMap::fitted_to_world(SEED, tall).sea_level(),
        HeightMap::fitted_to_world(SEED, short).sea_level()
    );
}

/// Property 2, over whole chunks: **the acceptance criterion.**
///
/// Two windows of the same seed, one 6 chunks tall and one 12, must produce
/// byte-identical chunks everywhere they overlap. Byte-identical means the
/// packed `u16`: block id, sunlight and block light.
#[test]
fn overlapping_windows_agree_byte_for_byte() {
    let (short_chunks, tall_chunks) = (6u32, 12u32);
    let mut short = WorldGen::new(spec(Carver::Sparse, short_chunks * 16));
    let mut tall = WorldGen::new(spec(Carver::Sparse, tall_chunks * 16));

    assert_eq!(short.bounds().len(), 6);
    assert_eq!(tall.bounds().len(), 12);

    let mut compared = 0;
    let mut non_air = 0;
    for cz in 0..2 {
        for cx in 0..2 {
            for cy in 0..short_chunks as i32 {
                let pos = ChunkPos::new(cx, cy, cz);
                let a = short.chunk_at(pos);
                let b = tall.chunk_at(pos);
                assert_eq!(
                    a.to_packed(),
                    b.to_packed(),
                    "chunk {pos:?} differs between a {short_chunks}-chunk and a \
                     {tall_chunks}-chunk window"
                );
                if !a.empty {
                    non_air += 1;
                }
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 4 * short_chunks as usize);
    assert!(
        non_air > 0,
        "every compared chunk was empty; the overlap test proved nothing"
    );
}

/// The overlap property for both carvers, since a carver that read the world
/// height would break it too.
#[test]
fn overlapping_windows_agree_for_every_carver() {
    for carver in [Carver::Density, Carver::Sparse] {
        let mut short = WorldGen::new(spec(carver, 64));
        let mut tall = WorldGen::new(spec(carver, 256));
        for cy in 0..4 {
            let pos = ChunkPos::new(1, cy, 1);
            assert_eq!(
                short.chunk_at(pos).to_packed(),
                tall.chunk_at(pos).to_packed(),
                "{carver} chunk {pos:?} differs across window heights"
            );
        }
    }
}

/// A window taller than the terrain band must add only air, and a window that
/// cuts into the band must not alter the chunks it does contain.
#[test]
fn growing_a_window_only_adds_chunks() {
    let mut small = WorldGen::new(spec(Carver::Density, 64));
    let mut big = WorldGen::new(spec(Carver::Density, 160));

    let in_band = ChunkPos::new(0, 1, 0);
    let a = small.chunk_at(in_band);
    let b = big.chunk_at(in_band);
    assert_eq!(a.to_packed(), b.to_packed(), "the shared band moved");

    // Above the small window's ceiling the big one has chunks the small one
    // does not, and they must be air with refreshed flags.
    let above = ChunkPos::new(0, 5, 0);
    let high = big.chunk_at(above);
    assert!(high.empty && !high.dirty, "a sky chunk must be culled, not dirty");
    assert!(
        high.blocks().iter().all(|&b| b == 0),
        "a chunk above the terrain band must be all air"
    );
}

/// The band must not have changed shape when the box coupling was removed:
/// every carver in the project was tuned against base 64 / amplitude 24.
#[test]
fn the_band_is_the_one_the_carvers_were_tuned_against() {
    use voxel_core::terrain::heightmap::{AMPLITUDE, BASE_Y, HeightMap};

    assert_eq!(BASE_Y, 64.0);
    assert_eq!(AMPLITUDE, 24.0);

    let hm = HeightMap::absolute(SEED);
    let (lo, hi) = (
        (BASE_Y - AMPLITUDE) as i32,
        (BASE_Y + AMPLITUDE) as i32,
    );
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    for x in -2048..2048 {
        let h = hm.height(x, 0);
        min = min.min(h);
        max = max.max(h);
        assert!((lo..=hi).contains(&h), "height {h} escaped {lo}..={hi}");
    }
    // Terrain must actually use the band, not sit in a corner of it.
    assert!(min < lo + 12, "the band is barely used at the bottom: min {min}");
    assert!(max > hi - 12, "the band is barely used at the top: max {max}");
}

/// Chunk-level independence: a given chunk's bytes must not depend on which
/// window it was generated in, even when the windows differ in more than height.
#[test]
fn a_chunk_is_the_same_in_any_window_that_contains_it() {
    let positions = [
        ChunkPos::new(0, 0, 0),
        ChunkPos::new(0, 1, 0),
        ChunkPos::new(0, 2, 0),
        ChunkPos::new(2, 1, 3),
    ];

    let mut reference = WorldGen::new(spec(Carver::Sparse, 64));
    let expected: Vec<Vec<u16>> = positions
        .iter()
        .map(|&p| reference.chunk_at(p).to_packed())
        .collect();

    // The same positions reached through windows of different heights, and
    // through a window whose band starts at the same place but is much taller.
    for height in [64u32, 80, 128, 512] {
        let mut other = WorldGen::new(spec(Carver::Sparse, height));
        for (i, &p) in positions.iter().enumerate() {
            assert_eq!(
                other.chunk_at(p).to_packed(),
                expected[i],
                "chunk {p:?} changed in a {height}-voxel window"
            );
        }
    }
}

/// Sanity: the voxel at a fixed world Y must be the same voxel in both windows,
/// which is what makes the overlap comparison meaningful rather than two empty
/// chunks agreeing.
#[test]
fn the_shared_band_actually_contains_terrain() {
    let mut streamer = WorldGen::new(spec(Carver::Sparse, 64));
    let c = streamer.chunk_at(ChunkPos::new(0, 1, 0));

    let mut solid = 0;
    for y in 0..16u8 {
        for z in 0..16u8 {
            for x in 0..16u8 {
                if !c.block(VoxelIndex::from_parts(x, y, z)).is_air() {
                    solid += 1;
                }
            }
        }
    }
    assert!(
        solid > 0,
        "the compared band is empty, so the overlap tests are vacuous"
    );
}
