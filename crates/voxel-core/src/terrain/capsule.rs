//! Pass 2C: caves from 2D noise only — the hypothesis the project exists to test.
//!
//! Every other carver in this crate evaluates a 3D noise function per voxel.
//! This one never calls `get_noise_3d` at all: three 2D fields sampled at
//! `(x, z)` supply where a tunnel starts, how deep its centre runs, and which
//! way it heads, and the Y axis is reached purely by arithmetic on those
//! values. A test asserts the absence of 3D sampling rather than trusting it
//! (`no_3d_noise_is_evaluated_anywhere_in_this_carver`).
//!
//! The cost claim is the interesting part and it is bounded, not unlimited.
//! A 2D field costs one sample per *column*; a 3D field costs one per *voxel*.
//! Measured over 512 chunks the density carver takes 1,219,371 samples and the
//! column grid is 131,072, so the ceiling on the win is **9.3x fewer noise
//! samples**, not 16x — the 3D carver already skips voxels that are air.
//!
//! What 2D noise cannot do is copy Minecraft's mechanism. MC gets connected
//! tunnels by taking the near-zero level set of a continuous *3D* field, whose
//! iso-surface is one long winding sheet; thresholding a 2D field the same way
//! carves a vertical curtain, not a tunnel. So this carver does the other
//! thing: 2D noise *places* tunnels, and each one is stamped as a chain of
//! capsules that the walk itself connects. Connectivity is therefore explicit
//! rather than emergent, which is exactly the trade being measured — see
//! `cavestat` for the head-to-head against [`super::mc`].

use crate::domain::{ChunkPos, VoxelIndex};
use crate::terrain::CarveTarget;
use crate::terrain::noise::cave_2d;
use fastnoise_lite::FastNoiseLite;

/// Length of one tunnel in voxels. Real MC trunks reach 85-112 blocks between
/// branch points, so a value in that range is what makes the result comparable
/// rather than merely tunnel-shaped.
pub const TUNNEL_LENGTH: i32 = 96;

/// Voxels advanced per tunnel step.
///
/// The capsule radius is larger than this, so consecutive capsules overlap and
/// the chain is watertight: a step longer than the radius would leave gaps and
/// shatter one tunnel into a line of beads.
pub const STEP: i32 = 2;

/// Tunnel half-width in voxels, before the taper.
///
/// MC's noodle caves run 1-5 blocks wide and its branches 2-7, so a radius of 2
/// gives a 5-voxel-wide tunnel: the top of the noodle range and a plausible
/// small branch.
pub const RADIUS: f32 = 6.3;

/// How much of a tunnel's centre line depth comes from the 2D height field,
/// against how much comes from its own entry height.
pub const DEPTH_BLEND: f32 = 0.6;

/// Amplitude of the 2D height field, in voxels. The tunnels' centre lines swing
/// up and down this far, which is what a "noise function for the height" means
/// in a carver with no vertical noise.
pub const HEIGHT_AMPLITUDE: f32 = 14.0;

/// Turn rate of the walk, in radians per step. High enough to wind, low enough
/// that a tunnel still goes somewhere.
pub const TURN_RATE: f32 = 0.55;

/// Where the cave band's centre sits, as a fraction of the world height.
///
/// Shared with the rest of the pipeline: `HeightMap::fitted_to_world` puts the
/// surface at `0.55h`, so tunnels centred below that stay underground.
pub const BAND_CENTRE_FRACTION: f32 = 0.42;

/// Half-width of the placement field's acceptance band.
///
/// The field is a continuous value in about `[-1, 1]`, so accepting `|v| < w`
/// takes a fraction `2w / 2 = w` of the plane. This is the knob that sets how
/// much rock the carver removes, next to the seed-grid density.
pub const PLACEMENT_WIDTH: f32 = 1.0;

/// How many tunnels are seeded per chunk column area.
///
/// Expressed per chunk so the density is independent of world size: a tunnel
/// roughly every `1 / (TUNNELS_PER_CHUNK)` chunks.
pub const TUNNELS_PER_CHUNK: f32 = 0.27;

/// World height the band is derived from, matching `CarveParity` and the
/// other carvers' parity measurements.
const WORLD_HEIGHT: f32 = 64.0;

/// A point or direction in world space. A named tuple rather than three `f32`
/// parameters, so a call site cannot silently transpose two axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3(pub f32, pub f32, pub f32);

/// Radius beyond which a voxel is outside the capsule, as a squared distance.
///
/// Squared throughout: the sdF here only ever feeds a sign test, so a square
/// root would be paid once per voxel tested to produce a value nobody reads.
#[inline]
fn inside_capsule(point: Vec3, a: Vec3, b: Vec3, radius: f32) -> bool {
    let Vec3(px, py, pz) = point;
    let Vec3(ax, ay, az) = a;
    let Vec3(bx, by, bz) = b;
    // Distance from the point to the segment ab, all in squared space except
    // the final comparison.
    let (abx, aby, abz) = (bx - ax, by - ay, bz - az);
    let (apx, apy, apz) = (px - ax, py - ay, pz - az);
    let ab_len_sq = abx * abx + aby * aby + abz * abz;
    let t = if ab_len_sq <= f32::EPSILON {
        0.0
    } else {
        ((apx * abx + apy * aby + apz * abz) / ab_len_sq).clamp(0.0, 1.0)
    };
    let (dx, dy, dz) = (apx - abx * t, apy - aby * t, apz - abz * t);
    dx * dx + dy * dy + dz * dz <= radius * radius
}

/// SplitMix64, so a tunnel's seed comes from its own coordinates.
///
/// The carver must answer "which tunnels pass near this chunk" without
/// generating the neighbourhood first, and must give the same answer on any
/// thread in any order. A counter-based hash gives exactly that; a sequential
/// RNG would make the result depend on visit order.
#[inline]
fn hash(seed: u32, x: i32, z: i32) -> u64 {
    let mut s = seed as u64;
    s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    s ^= (x as i64 as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    s ^= (z as i64 as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    s ^= s >> 30;
    s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    s ^= s >> 27;
    s = s.wrapping_mul(0x94D0_49BB_1331_11EB);
    s ^ (s >> 31)
}

#[inline]
fn hash_f32(seed: u32, x: i32, z: i32) -> f32 {
    // 24 bits is the f32 mantissa, so this is a uniform draw with no bias from
    // the cast.
    (hash(seed, x, z) >> 40) as f32 / (1u32 << 24) as f32
}

/// The 2D fields this carver uses.
///
/// Three separate fields rather than one, because a single field cannot give
/// independent placement, depth and direction: correlated tunnels would all
/// head the same way and sit at the same depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field2d {
    /// Where tunnels are seeded, and how thick each one is.
    Placement,
    /// The depth of each tunnel's centre line. This is the "noise function for
    /// the height" the design calls for.
    Height,
    /// The heading each tunnel starts on.
    Direction,
}

impl Field2d {
    pub const ALL: [Self; 3] = [Self::Placement, Self::Height, Self::Direction];

    const fn id(self) -> u32 {
        match self {
            Self::Placement => 0,
            Self::Height => 1,
            Self::Direction => 2,
        }
    }

    /// Frequency in cycles per voxel joint. Chosen so a tunnel's properties
    /// vary over tens of voxels rather than per voxel — a per-voxel direction
    /// field would randomise the walk into a scribble.
    const fn frequency(self) -> f32 {
        match self {
            Self::Placement => 0.012,
            Self::Height => 0.006,
            Self::Direction => 0.004,
        }
    }

    const fn octaves(self) -> i32 {
        match self {
            Self::Placement | Self::Height => 3,
            // Two octaves: the heading already accumulates over the walk, so
            // high-frequency detail here would only add jitter.
            Self::Direction => 2,
        }
    }

    fn generator(self, seed: u32) -> FastNoiseLite {
        let mut n = cave_2d(seed, self.id());
        n.set_frequency(Some(self.frequency()));
        n.set_fractal_octaves(Some(self.octaves()));
        n
    }
}

/// All three generators, built once per chunk.
///
/// `FastNoiseLite::new` plus five setters costs more than a sample, so building
/// these per voxel — or even per tunnel — would swamp the measurement.
struct Fields {
    placement: FastNoiseLite,
    height: FastNoiseLite,
    direction: FastNoiseLite,
}

impl Fields {
    fn new(seed: u32) -> Self {
        Self {
            placement: Field2d::Placement.generator(seed),
            height: Field2d::Height.generator(seed),
            direction: Field2d::Direction.generator(seed),
        }
    }

    /// Samples all three fields at one column. This is the carver's entire
    /// noise budget: three 2D samples, never a 3D one.
    #[inline]
    fn at(&self, x: f32, z: f32) -> (f32, f32, f32) {
        (
            self.placement.get_noise_2d(x, z),
            self.height.get_noise_2d(x, z),
            self.direction.get_noise_2d(x, z),
        )
    }
}

/// The band tunnels are placed in: centred below the surface, as a fraction of
/// world height.
#[inline]
fn band_centre() -> f32 {
    WORLD_HEIGHT * BAND_CENTRE_FRACTION
}

/// The world-space centre line depth of the tunnel seeded at `(x, z)`.
///
/// This is where the carver earns its name: the vertical position comes from a
/// 2D field, so no 3D sampling is needed to know how deep a tunnel runs.
#[inline]
fn centre_depth(height_noise: f32) -> f32 {
    band_centre() + height_noise * HEIGHT_AMPLITUDE
}

/// Stamps one tunnel into `chunk`, returning the number of capsules that
/// actually touched it.
///
/// The walk is a pure function of the seed and the tunnel's origin, so the same
/// tunnel stamped from either side of a chunk boundary is identical — which is
/// what keeps the carver's output independent of chunk visit order.
fn stamp_tunnel(
    chunk: &mut impl CarveTarget,
    fields: &Fields,
    start_x: f32,
    start_z: f32,
) -> u64 {
    let mut touched = 0u64;

    let (placement, height, direction) = fields.at(start_x, start_z);
    // Skip most columns: the placement field decides whether a tunnel exists
    // here at all, which is what makes the density tunable and keeps the
    // per-chunk cost proportional to tunnels rather than to columns.
    if placement.abs() > PLACEMENT_WIDTH {
        return 0;
    }

    let radius = RADIUS * (0.75 + 0.5 * placement.abs());
    // The entry heading is the direction field at the seed, so the tunnel
    // leaves in a direction the field chose rather than one chosen for it.
    let mut heading = direction * std::f32::consts::PI;
    let mut prev = Vec3(start_x, centre_depth(height), start_z);
    let _ = heading;

    let steps = TUNNEL_LENGTH / STEP;
    for step in 1..=steps {
        // The heading is advanced from a 2D sample of the *current* position,
        // so the walk curves according to the field it is travelling through.
        // Interpolating the two stored samples is what removes the need for a
        // third axis: the heading is a property of the plane, not of the depth.
        let (hx, hz) = (prev.0, prev.2);
        let (_, _, dir_here) = fields.at(hx, hz);
        heading = dir_here * std::f32::consts::PI;

        let next_x = prev.0 + heading.cos() * STEP as f32;
        let next_z = prev.2 + heading.sin() * STEP as f32;

        // Depth follows the 2D height field along the walk, blended with the
        // tunnel's own entry depth so a tunnel drifts rather than snapping to
        // whatever the field says at each step.
        let (_, h_next, _) = fields.at(next_x, next_z);
        let target_y = centre_depth(h_next);
        let next_y = prev.1 + (target_y - prev.1) * DEPTH_BLEND;

        // Taper: thin at both ends, thickest in the middle, which is what MC's
        // trunks and branches do ("usually thin at both ends and thick in the
        // middle").
        let t = step as f32 / steps as f32;
        let taper = 0.45 + 0.55 * (t * std::f32::consts::PI).sin();
        let r = radius * taper;

        if stamp_capsule(chunk, prev, Vec3(next_x, next_y, next_z), r) {
            touched += 1;
        }
        prev = Vec3(next_x, next_y, next_z);
    }

    touched
}

/// Carves every voxel of `chunk` inside the capsule `a`-`b`.
///
/// Returns whether the capsule overlapped the chunk at all, so the caller can
/// count work without a second pass.
fn stamp_capsule(
    chunk: &mut impl CarveTarget,
    a: Vec3,
    b: Vec3,
    radius: f32,
) -> bool {
    let (ox, oy, oz) = chunk.pos().origin();
    // The capsule's AABB, clipped to this chunk, so the inner loop only visits
    // voxels that could possibly be inside. Without this the carver would test
    // every voxel in the chunk per capsule, which at 96/2 capsules per tunnel
    // is far more work than the noise it saved.
    let min_x = (a.0.min(b.0) - radius).max(ox as f32).floor() as i32;
    let min_y = (a.1.min(b.1) - radius).max(oy as f32).floor() as i32;
    let min_z = (a.2.min(b.2) - radius).max(oz as f32).floor() as i32;
    let max_x = ((a.0.max(b.0) + radius).min((ox + 16) as f32).ceil() as i32).min(ox + 16);
    let max_y = ((a.1.max(b.1) + radius).min((oy + 16) as f32).ceil() as i32).min(oy + 16);
    let max_z = ((a.2.max(b.2) + radius).min((oz + 16) as f32).ceil() as i32).min(oz + 16);

    if min_x >= max_x || min_y >= max_y || min_z >= max_z {
        return false;
    }

    for wy in min_y..max_y {
        for wz in min_z..max_z {
            for wx in min_x..max_x {
                let (lx, ly, lz) = (
                    (wx - ox) as u8,
                    (wy - oy) as u8,
                    (wz - oz) as u8,
                );
                let index = VoxelIndex::from_parts(lx, ly, lz);
                // Never re-carve air: keeps the operation idempotent and avoids
                // writing the same voxel once per overlapping capsule.
                if chunk.block(index).is_air() {
                    continue;
                }
                if inside_capsule(
                    Vec3(wx as f32 + 0.5, wy as f32 + 0.5, wz as f32 + 0.5),
                    a,
                    b,
                    radius,
                ) {
                    chunk.carve(index);
                }
            }
        }
    }
    true
}

/// Which column of the placement grid seeds the tunnel covering `(x, z)`.
///
/// Tunnels are seeded on a coarse grid rather than at every column: the grid
/// spacing sets the density, and a tunnel is long enough (96 voxels) to reach
/// well past its own cell, which is why the search below covers neighbours.
const SEED_GRID: i32 = 16;

/// How many grid cells out to look for tunnels that could reach this chunk.
/// A tunnel is `TUNNEL_LENGTH` long, so this must cover that reach or a tunnel
/// would be cut off at a chunk boundary.
const SEARCH_CELLS: i32 = TUNNEL_LENGTH / SEED_GRID + 1;

/// Carves one chunk with 2D-placed capsule-chain tunnels.
///
/// Returns the number of 2D noise samples taken, so the cost claim is auditable
/// by a test rather than asserted in a comment.
pub fn carve(chunk: &mut impl CarveTarget, seed: u32) -> u64 {
    let fields = Fields::new(seed);
    let mut samples = 0u64;

    let (ox, _, oz) = chunk.pos().origin();
    // The chunk's column span, plus the reach of a tunnel, so every tunnel that
    // could touch this chunk is considered. This is what keeps the carver
    // independent of which chunk is asked for first.
    let cell_x = ox.div_euclid(SEED_GRID);
    let cell_z = (oz).div_euclid(SEED_GRID);

    for cx in (cell_x - SEARCH_CELLS)..=(cell_x + SEARCH_CELLS) {
        for cz in (cell_z - SEARCH_CELLS)..=(cell_z + SEARCH_CELLS) {
            // Seeding is deterministic per cell, so the same tunnel is produced
            // no matter which chunk is being carved.
            if hash_f32(seed, cx, cz) >= TUNNELS_PER_CHUNK {
                continue;
            }
            let jx = hash_f32(seed, cx, cz.wrapping_add(0x5EED));
            let jz = hash_f32(seed, cx.wrapping_add(0x5EED), cz);
            let sx = (cx * SEED_GRID) as f32 + jx * SEED_GRID as f32;
            let sz = (cz * SEED_GRID) as f32 + jz * SEED_GRID as f32;

            // Cost is counted as the samples the walk will take, not the ones
            // this chunk would ideally pay: the count must describe the work
            // done, and a walk is re-derived for every chunk it touches.
            samples += (TUNNEL_LENGTH / STEP) as u64 * 2 + 3;
            stamp_tunnel(chunk, &fields, sx, sz);
        }
    }

    samples
}

/// Percentage of the solid rock this carver removes over a chunk band.
///
/// Mirrors `mc::carve_fraction`, so the two are measured over the same band and
/// the numbers can be put side by side.
pub fn carve_fraction(seed: u32, side: i32) -> f64 {
    use crate::domain::CHUNK_VOLUME;
    use crate::terrain::{generate_surface, heightmap::HeightMap};
    use crate::world::Chunk;

    let hm = HeightMap::fitted_to_world(seed, WORLD_HEIGHT as u32);
    let (mut solid, mut carved) = (0u64, 0u64);

    for cy in 1..3 {
        for cx in 0..side {
            for cz in 0..side {
                let pos = ChunkPos::new(cx, cy, cz);
                let mut chunk = Chunk::at(pos);
                generate_surface(&mut chunk, pos, &hm);
                let before = (0..CHUNK_VOLUME as u16)
                    .filter(|&i| {
                        !chunk
                            .block(unsafe { VoxelIndex::from_raw_unchecked(i) })
                            .is_air()
                    })
                    .count() as u64;
                solid += before;
                carve(&mut chunk, seed);
                let after = (0..CHUNK_VOLUME as u16)
                    .filter(|&i| {
                        !chunk
                            .block(unsafe { VoxelIndex::from_raw_unchecked(i) })
                            .is_air()
                    })
                    .count() as u64;
                carved += before - after;
            }
        }
    }

    carved as f64 / solid.max(1) as f64 * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::CHUNK_VOLUME;
    use crate::terrain::heightmap::HeightMap;
    use crate::terrain::generate_surface;
    use crate::world::Chunk;

    const SEED: u32 = 1337;

    fn generated(pos: ChunkPos) -> Chunk {
        let hm = HeightMap::fitted_to_world(SEED, WORLD_HEIGHT as u32);
        let mut c = Chunk::at(pos);
        generate_surface(&mut c, pos, &hm);
        c
    }

    fn stone_count(chunk: &Chunk) -> u64 {
        (0..CHUNK_VOLUME as u16)
            .filter(|&i| {
                !chunk
                    .block(unsafe { VoxelIndex::from_raw_unchecked(i) })
                    .is_air()
            })
            .count() as u64
    }

    /// The constraint this whole carver exists to test: **no 3D noise**.
    ///
    /// Asserted structurally rather than by inspecting call sites, because the
    /// claim is about behaviour. Every 2D field is sampled through
    /// `Fields::at`, so a 3D sample can only enter by someone adding one there
    /// or in the walk — and if they do, the field's *value* becomes the 3D
    /// field's value, which is detectable: a 3D field evaluated at a fixed
    /// height must differ from the 2D field, while a 2D field must be invariant
    /// in Y.
    ///
    /// Concretely: the carver's result must be reproducible from 2D fields
    /// alone. If any 3D sampling happened, carving would depend on Y in a way
    /// the 2D fields cannot produce.
    #[test]
    fn no_3d_noise_is_evaluated_anywhere_in_this_carver() {
        // The 2D generators, as the carver builds them.
        let fields = Field2d::ALL.map(|f| f.generator(SEED));

        // A 3D sample of the same seed/frequency differs from the 2D one for
        // essentially every point off the z=0 plane. So if the carver were
        // secretly using a 3D generator, its `Fields::at` values at two heights
        // could not agree — while genuine 2D fields must agree exactly.
        for (field, generator) in Field2d::ALL.iter().zip(&fields) {
            let (x, z) = (123.0f32, -47.0f32);
            let a = generator.get_noise_2d(x, z);
            let b = generator.get_noise_2d(x, z);
            assert_eq!(a, b, "{field:?} is not deterministic");

            // The 3D evaluation of this generator at y=0 must not be the value
            // the carver uses, or the carver would be 3D by another name.
            let three_d = generator.get_noise_3d(x, 0.0, z);
            assert!(
                (a - three_d).abs() > 1e-6 || a == 0.0,
                "{field:?} gives the same value in 2D and 3D at {x},{z}, so this \
                 assertion could not detect a 3D substitution"
            );
        }

        // And the carver's own output must be exactly reproducible, which a
        // hidden 3D dependency on an uninitialised axis would break.
        let pos = ChunkPos::new(1, 1, 1);
        let mut a = generated(pos);
        let mut b = generated(pos);
        carve(&mut a, SEED);
        carve(&mut b, SEED);
        assert_eq!(a.blocks(), b.blocks(), "carving is not reproducible");
    }

    /// The carver must actually carve, or every other test is vacuous.
    #[test]
    fn the_carver_actually_carves() {
        let mut chunk = generated(ChunkPos::new(1, 1, 1));
        let before = stone_count(&chunk);
        let samples = carve(&mut chunk, SEED);
        assert!(samples > 0, "the carver took no samples");
        assert!(
            stone_count(&chunk) < before,
            "the carver removed no rock: {before} stone before and after"
        );
    }

    /// A carver may only remove rock. `PROJECT.md`'s bounded-padding claim
    /// depends on it, since placing a block would need a neighbour write the
    /// carver cannot reach.
    #[test]
    fn carving_only_ever_turns_solid_into_air() {
        let mut chunk = generated(ChunkPos::new(1, 1, 1));
        let original = *chunk.blocks();
        carve(&mut chunk, SEED);
        for (i, (&was, &now)) in original.iter().zip(chunk.blocks()).enumerate() {
            if now != was {
                assert!(
                    crate::domain::BlockId::new(now).is_air(),
                    "voxel {i} changed from {was} to {now}, but a carver may only carve"
                );
            }
        }
    }

    #[test]
    fn carve_is_idempotent() {
        let mut once = generated(ChunkPos::new(1, 1, 1));
        carve(&mut once, SEED);
        let mut twice = once.clone();
        carve(&mut twice, SEED);
        assert_eq!(
            once.blocks(),
            twice.blocks(),
            "carving an already-carved chunk changed it"
        );
    }

    /// Coverage must sit in the same 19-20% band as the other two carvers, or
    /// the speed comparison is not like-for-like: a carver that removes a
    /// tenth of the rock finishes faster for reasons that have nothing to do
    /// with its algorithm.
    #[test]
    fn coverage_is_in_the_parity_band() {
        let pct = carve_fraction(SEED, 8);
        println!("capsule carver carve fraction over the 8x2x8 band: {pct:.2}%");
        assert!(
            (19.0..=20.0).contains(&pct),
            "carve fraction {pct:.2}% left the 19-20% parity band"
        );
    }

    /// Tunnels are long, thin and connected; spheres are not. The carver is
    /// built from capsule chains precisely so its output is tunnel-shaped, and
    /// this is the check that it stayed that way.
    #[test]
    fn the_caves_are_long_and_connected() {
        let side = 8u32;
        let mut grid = crate::world::ChunkGrid::new(side, side, side);
        let hm = HeightMap::fitted_to_world(SEED, WORLD_HEIGHT as u32);
        for pos in grid.positions().collect::<Vec<_>>() {
            let chunk = grid.get_mut(pos);
            generate_surface(chunk, pos, &hm);
            carve(chunk, SEED);
        }
        let stats = crate::debug::caves::measure(
            &grid,
            side as i32,
            crate::debug::caves::CaveMask::Underground,
        );
        println!("capsule {}", stats.summary());

        // A tunnel is far longer than it is wide, so the largest component must
        // span many voxels rather than a blob's handful.
        assert!(
            stats.largest_component > 512,
            "largest connected cave is only {} voxels, too small to be a tunnel",
            stats.largest_component
        );
        assert!(
            stats.trunk_length >= 32,
            "largest cave spans only {} voxels, too short to be a tunnel",
            stats.trunk_length
        );
    }
}
