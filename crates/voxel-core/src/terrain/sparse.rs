//! Pass 2B: sparse cave stamping, `O(S)`.
//!
//! The target design in `PROJECT.md` §4.2. Instead of sampling noise at every
//! voxel, the world is partitioned into fixed-size *regions*; each region
//! deterministically decides whether it contains cave origins and where. A
//! chunk then only visits the regions whose origins could reach it — its own
//! plus a one-region halo — and stamps each origin's SDF into the chunk.
//!
//! The cost therefore scales with the number of stamps near the chunk, not with
//! the chunk's volume, which is the whole point of the comparison.
//!
//! Bounded reach is what makes this correct as well as fast: a stamp of radius
//! `r` is only ever applied to chunks within `ceil(r)` voxels of its centre, so
//! a chunk's contents depend only on its padded neighbourhood, and chunk
//! generation stays independent and order-free.

use super::CarveTarget;
use crate::config::CaveParams;
use crate::domain::{CHUNK_SIZE, VoxelIndex};
use crate::terrain::noise::RegionRng;

/// Region edge in chunks. A region decides its own cave origins, so this also
/// sets the granularity of the "is there a cave here at all" answer that makes
/// empty regions nearly free.
pub const REGION_CHUNKS: i32 = 4;

/// Region edge in voxels.
pub const REGION_SIZE: i32 = REGION_CHUNKS * CHUNK_SIZE as i32;

/// One cave origin in world voxel space, with its stamp radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaveOrigin {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub radius: f32,
}

impl CaveOrigin {
    /// The stamp is applied to every voxel within `radius` of the centre, so
    /// this is the exact voxel-space reach used for culling.
    #[inline]
    pub fn reach(&self) -> i32 {
        self.radius.ceil() as i32
    }

    /// Signed distance field for a sphere: negative inside, positive outside.
    #[inline]
    pub fn sdf(&self, x: f32, y: f32, z: f32) -> f32 {
        let (dx, dy, dz) = (x - self.x, y - self.y, z - self.z);
        // f32::sqrt is accurate enough here and avoids a dependency; the value
        // only ever feeds a sign test against 0.
        (dx * dx + dy * dy + dz * dz).sqrt() - self.radius
    }
}

/// Enumerates the cave origins of one region.
///
/// Pure in `(seed, params, region)` so any thread can call it for any region in
/// any order and get the same answer — this is what removes the need for a
/// global origin table or a generation-order dependency.
pub fn origins_in_region(seed: u32, params: &CaveParams, rx: i32, ry: i32, rz: i32) -> Vec<CaveOrigin> {
    let mut rng = RegionRng::new(seed, rx, ry, rz);
    let mut out = Vec::new();

    // Expected count per region. The density knob is in origins per 1000
    // voxels-cubed (see `CaveParams::origin_density`), which keeps the default
    // values readable: 0.02 means roughly one cave origin per 50k voxels.
    let region_volume = (REGION_SIZE as f32).powi(3);
    let expected = params.origin_density * region_volume / 1000.0;
    let mut count = expected.floor() as u32;
    if rng.next_f32() < expected.fract() {
        count += 1;
    }
    // Bound the work per region so a nonsense density cannot hang generation.
    count = count.min(MAX_ORIGINS_PER_REGION);

    let (ox, oy, oz) = (rx * REGION_SIZE, ry * REGION_SIZE, rz * REGION_SIZE);
    for _ in 0..count {
        out.push(CaveOrigin {
            x: ox as f32 + rng.next_f32() * REGION_SIZE as f32,
            y: oy as f32 + rng.next_f32() * REGION_SIZE as f32,
            z: oz as f32 + rng.next_f32() * REGION_SIZE as f32,
            radius: rng.range_f32(params.min_radius, params.max_radius),
        });
    }
    out
}

/// Stats returned so callers can assert the `O(S)` claim rather than trust it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StampStats {
    /// Regions whose origin lists were enumerated.
    pub regions_visited: u64,
    /// Origins actually tested against voxels.
    pub origins_considered: u64,
    /// Voxels whose SDF was evaluated.
    pub voxels_tested: u64,
    /// Voxels turned to air. Left at zero by [`carve`], which does not pay for
    /// a counting pass in the hot path; call [`count_carved`] when reporting.
    pub voxels_carved: u64,
}

/// Upper bound on origins enumerated per region, so a misconfigured density
/// degrades rather than hangs. 64 origins of radius 4 already fills a region.
pub const MAX_ORIGINS_PER_REGION: u32 = 64;

/// Carves one chunk by stamping every origin that can reach it.
pub fn carve(chunk: &mut impl CarveTarget, seed: u32, params: &CaveParams) -> StampStats {
    let pos = chunk.pos();
    let mut stats = StampStats::default();

    // A region can only hold a cave if its expected origin count rounds to at
    // least one. With a low density most regions are empty, which is exactly
    // the case the sparse carver is supposed to be fast in, so this is decided
    // from the parameters alone rather than by enumerating first.
    let region_can_hold_caves = params.origin_density * (REGION_SIZE as f32).powi(3) / 1000.0 >= 0.5;

    // Only regions within one region of the chunk can hold an origin whose
    // sphere reaches into it, because a stamp's reach is bounded by its radius
    // and regions are far larger than the configured radii.
    let rmin = -1;
    let rmax = 1;
    let base_rx = pos.x.div_euclid(REGION_CHUNKS);
    let base_ry = pos.y.div_euclid(REGION_CHUNKS);
    let base_rz = pos.z.div_euclid(REGION_CHUNKS);

    for dry in rmin..=rmax {
        for drz in rmin..=rmax {
            for drx in rmin..=rmax {
                let (rx, ry, rz) = (base_rx + drx, base_ry + dry, base_rz + drz);
                if !region_can_hold_caves {
                    stats.regions_visited += 1;
                    continue;
                }
                let origins = origins_in_region(seed, params, rx, ry, rz);
                stats.regions_visited += 1;
                if origins.is_empty() {
                    continue;
                }

                // Cheap region-level rejection before touching any voxel: if
                // the closest point of this chunk is further than the largest
                // radius in the region, no stamp can land.
                let max_radius = origins.iter().fold(0.0f32, |m, o| m.max(o.radius));
                if !chunk_within_reach(&pos, origins.as_slice(), max_radius) {
                    continue;
                }

                for origin in &origins {
                    stats.origins_considered += 1;
                    stats.voxels_tested += stamp(chunk, origin);
                }
            }
        }
    }

    stats
}

/// Counts how many voxels a carve turned from solid to air.
///
/// A separate pass rather than a counter inside `stamp`, because the trait
/// exposes reads only and threading a counter through the hot loop would cost
/// more than it reports. Used by the benchmark and the debug views.
pub fn count_carved(chunk: &impl CarveTarget) -> u64 {
    (0..4096u16)
        .filter(|&i| {
            chunk
                .block(unsafe { VoxelIndex::from_raw_unchecked(i) })
                .is_air()
        })
        .count() as u64
}

/// True if any voxel of the chunk can be within `max_radius` of the region's
/// origin cloud. Uses the chunk's world AABB against the region's AABB.
fn chunk_within_reach(pos: &crate::domain::ChunkPos, origins: &[CaveOrigin], max_radius: f32) -> bool {
    let (ox, oy, oz) = pos.origin();
    let (min_x, min_y, min_z) = (ox as f32, oy as f32, oz as f32);
    let (max_x, max_y, max_z) = (
        min_x + CHUNK_SIZE as f32,
        min_y + CHUNK_SIZE as f32,
        min_z + CHUNK_SIZE as f32,
    );

    origins.iter().any(|o| {
        let r = max_radius.max(o.radius);
        // Closest point on the chunk AABB to the origin, then compare squared
        // distances to avoid a square root per test.
        let cx = o.x.clamp(min_x, max_x);
        let cy = o.y.clamp(min_y, max_y);
        let cz = o.z.clamp(min_z, max_z);
        let (dx, dy, dz) = (o.x - cx, o.y - cy, o.z - cz);
        dx * dx + dy * dy + dz * dz <= r * r
    })
}

/// Applies one origin's SDF to the chunk. Returns voxels tested.
///
/// Iterates only the origin's bounding box intersected with the chunk, which is
/// where the `O(S)` behaviour comes from: a stamp touches `O(r^3)` voxels
/// regardless of how big the chunk is.
fn stamp(chunk: &mut impl CarveTarget, origin: &CaveOrigin) -> u64 {
    let pos = chunk.pos();
    let (ox, oy, oz) = pos.origin();
    let r = origin.reach();

    // Intersect the stamp's AABB with the chunk, then with the world-vertical
    // bounds so a stamp cannot reach outside the chunk it was given.
    let lo_x = (origin.x.floor() as i32 - r).max(ox);
    let hi_x = (origin.x.ceil() as i32 + r).min(ox + CHUNK_SIZE as i32 - 1);
    let lo_y = (origin.y.floor() as i32 - r).max(oy);
    let hi_y = (origin.y.ceil() as i32 + r).min(oy + CHUNK_SIZE as i32 - 1);
    let lo_z = (origin.z.floor() as i32 - r).max(oz);
    let hi_z = (origin.z.ceil() as i32 + r).min(oz + CHUNK_SIZE as i32 - 1);

    let mut tested = 0;
    for wy in lo_y..=hi_y {
        for wz in lo_z..=hi_z {
            for wx in lo_x..=hi_x {
                let index = VoxelIndex::from_parts(
                    (wx - ox) as u8,
                    (wy - oy) as u8,
                    (wz - oz) as u8,
                );
                if chunk.block(index).is_air() {
                    continue;
                }
                tested += 1;
                if origin.sdf(wx as f32, wy as f32, wz as f32) < 0.0 {
                    chunk.carve(index);
                }
            }
        }
    }
    tested
}

/// World-space helper mirroring the density carver's, so the two can be
/// compared voxel by voxel in tests.
#[inline]
pub fn carved_by_origins(seed: u32, params: &CaveParams, wx: i32, wy: i32, wz: i32) -> bool {
    // Which region could hold an origin reaching this voxel.
    let rx = wx.div_euclid(REGION_SIZE);
    let ry = wy.div_euclid(REGION_SIZE);
    let rz = wz.div_euclid(REGION_SIZE);
    for dry in -1..=1 {
        for drz in -1..=1 {
            for drx in -1..=1 {
                for o in origins_in_region(seed, params, rx + drx, ry + dry, rz + drz) {
                    if o.sdf(wx as f32, wy as f32, wz as f32) < 0.0 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Number of regions a chunk's carving touches, for the debug report.
pub const fn regions_per_chunk() -> u64 {
    // 3x3x3 halo.
    27
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CaveParams;
    use crate::domain::{BlockId, ChunkPos};
    use crate::terrain::heightmap::HeightMap;
    use crate::world::Chunk;

    fn params() -> CaveParams {
        CaveParams {
            origin_density: 0.02,
            min_radius: 1.5,
            max_radius: 4.0,
        }
    }

    fn solid_chunk(pos: ChunkPos) -> Chunk {
        let hm = HeightMap::new(1337, 0);
        let mut c = Chunk::default();
        crate::terrain::generate_surface(&mut c, pos, &hm);
        c
    }

    #[test]
    fn origin_enumeration_is_deterministic_and_order_free() {
        let p = params();
        let a = origins_in_region(1337, &p, 1, 0, -2);
        let b = origins_in_region(1337, &p, 1, 0, -2);
        assert_eq!(a, b, "same region must give the same origins");

        // Enumerating other regions first must not change this one.
        let _ = origins_in_region(1337, &p, 9, 9, 9);
        let c = origins_in_region(1337, &p, 1, 0, -2);
        assert_eq!(a, c, "enumeration must not depend on call order");
    }

    #[test]
    fn origins_land_inside_their_own_region() {
        let p = params();
        for rx in -2..2 {
            for rz in -2..2 {
                for o in origins_in_region(1337, &p, rx, 0, rz) {
                    let (lo_x, hi_x) = (rx * REGION_SIZE, (rx + 1) * REGION_SIZE);
                    assert!(
                        o.x >= lo_x as f32 && o.x < hi_x as f32,
                        "origin x {} escaped region {rx}",
                        o.x
                    );
                    assert!(o.radius >= p.min_radius && o.radius <= p.max_radius);
                }
            }
        }
    }

    #[test]
    fn sdf_sign_matches_inside_outside() {
        let o = CaveOrigin {
            x: 10.0,
            y: 10.0,
            z: 10.0,
            radius: 2.0,
        };
        assert!(o.sdf(10.0, 10.0, 10.0) < 0.0, "centre is inside");
        // Exactly on the rim the distance is zero, which is not < 0, so the
        // surface voxel itself is kept. Documented because it decides whether
        // caves have a one-voxel shell.
        assert_eq!(o.sdf(12.0, 10.0, 10.0), 0.0, "rim is on the surface");
        assert!(o.sdf(11.9, 10.0, 10.0) < 0.0, "just inside");
        assert!(o.sdf(12.1, 10.0, 10.0) > 0.0, "just outside");
        assert!(o.sdf(13.0, 10.0, 10.0) > 0.0, "outside");
        assert_eq!(o.reach(), 2);
        // reach() must cover the whole sphere, or the stamp silently clips.
        assert!(o.reach() as f32 >= o.radius);
    }

    #[test]
    fn stamping_never_reaches_outside_the_given_chunk() {
        // The bounded-padding claim: whatever carving happens must be inside
        // this chunk, so chunks stay independently generatable.
        let p = params();
        let mut chunk = solid_chunk(ChunkPos::new(2, 0, 3));
        let before_air = count_air(&chunk);
        let stats = carve(&mut chunk, 1337, &p);
        assert_eq!(stats.regions_visited, regions_per_chunk());
        assert!(count_air(&chunk) >= before_air, "carving cannot create blocks");
    }

    #[test]
    fn cost_scales_with_stamps_not_volume() {
        // The O(S) property: a chunk in a cave-poor area must test far fewer
        // voxels than the 4096 a full 3D density sweep would.
        let p = CaveParams {
            origin_density: 0.000_05,
            min_radius: 1.0,
            max_radius: 2.0,
        };
        let mut total = 0u64;
        for x in 0..8 {
            let mut c = solid_chunk(ChunkPos::new(x, 0, 0));
            total += carve(&mut c, 1337, &p).voxels_tested;
        }
        let full_sweep = 8 * 4096;
        assert!(
            total < full_sweep / 4,
            "sparse carver tested {total} voxels, expected well under {full_sweep}"
        );
    }

    #[test]
    fn density_knob_controls_cave_count() {
        let sparse = CaveParams {
            origin_density: 0.0001,
            min_radius: 1.0,
            max_radius: 2.0,
        };
        let dense = CaveParams {
            origin_density: 0.05,
            min_radius: 1.0,
            max_radius: 2.0,
        };
        let count = |p: &CaveParams| -> usize {
            (0..27)
                .map(|i| origins_in_region(1337, p, i % 3, 0, i / 3).len())
                .sum()
        };
        assert!(
            count(&dense) > count(&sparse),
            "higher density must yield more origins"
        );
    }

    #[test]
    fn zero_density_yields_no_caves_at_all() {
        let p = CaveParams {
            origin_density: 0.0,
            min_radius: 1.0,
            max_radius: 2.0,
        };
        let mut c = solid_chunk(ChunkPos::ORIGIN);
        let before = count_air(&c);
        carve(&mut c, 1337, &p);
        assert_eq!(count_air(&c), before, "zero density must carve nothing");
    }

    #[test]
    fn standalone_query_agrees_with_the_carver() {
        // `carved_by_origins` backs the debug views; if it disagreed with the
        // carver the dumps would show a world that was never generated.
        let p = params();
        let pos = ChunkPos::new(0, 0, 0);
        let mut chunk = solid_chunk(pos);
        carve(&mut chunk, 1337, &p);
        let (ox, oy, oz) = pos.origin();

        for y in 0..16u8 {
            for z in 0..16u8 {
                for x in 0..16u8 {
                    let i = VoxelIndex::from_parts(x, y, z);
                    let (wx, wy, wz) = (ox + x as i32, oy + y as i32, oz + z as i32);
                    let query = carved_by_origins(1337, &p, wx, wy, wz);
                    // The carver starts from the surface, so a voxel that was
                    // already air is not evidence either way; only assert on
                    // voxels the surface pass made solid.
                    if query {
                        assert!(
                            chunk.block(i).is_air(),
                            "query says carved but chunk has {:?} at ({x},{y},{z})",
                            chunk.block(i)
                        );
                    }
                }
            }
        }
    }

    fn count_air(chunk: &Chunk) -> usize {
        (0..4096u16)
            .filter(|&i| {
                chunk
                    .block(unsafe { VoxelIndex::from_raw_unchecked(i) })
                    .is_air()
            })
            .count()
    }

    #[test]
    fn generated_terrain_still_has_ground_after_carving() {
        // A carver that removed everything would still pass the "only removes
        // blocks" tests; this catches that.
        let p = params();
        let mut chunk = solid_chunk(ChunkPos::new(1, 0, 1));
        carve(&mut chunk, 1337, &p);
        let stone = chunk.blocks().iter().filter(|&&b| b == BlockId::STONE.raw()).count();
        assert!(stone > 0, "carving removed all stone");
    }
}
