//! Pass 1: 2D surface heightmap.
//!
//! Sampled once per world column and shared by both Pass 2 strategies, so the
//! benchmark compares cave carving rather than terrain setup. Pure function of
//! `(seed, x, z)`, which is what lets chunk generation run in any order on any
//! thread and still agree.
//!
//! ## Absolute Y, not box-relative
//!
//! The band is a fixed range of **absolute world Y** ([`BASE_Y`] ±
//! [`AMPLITUDE`]), the way Minecraft's overworld has a fixed `-56..180` span.
//! That is the whole point: `(seed, x, z) -> height` must not mention the world
//! size, or two runs of the same seed at different heights describe different
//! worlds and cannot be compared or clipped.
//!
//! This replaces `fitted_to_world(seed, world_height)`, which derived base,
//! amplitude and sea level from the box (`h * 0.55`, `h * 0.18`, `h * 0.5`). It
//! was written to stop a short world generating as solid stone, and it did — at
//! the cost of making terrain a function of world *size*. A world height is now
//! purely a clipping window: the same world, with Y outside the window simply
//! not materialised.
//!
//! The old shape also meant a world under 32 voxels tall had `sea_level` at or
//! above its own ceiling, so no column reached the top of a chunk and
//! `Chunk::enclosed` could never be set, silently losing solid-rock culling.
//! Sea level is now absolute too, so that cannot recur: it is either inside a
//! window or it is not, and `sea_level_is_absolute_not_derived_from_the_window`
//! asserts it.

use super::noise;

/// Mean surface height, in absolute world Y.
///
/// 64 is the same centre the old `new(seed, sea_level)` used for its base, so
/// terrain that was already tuned against `HeightMap::new` keeps its shape.
pub const BASE_Y: f32 = 64.0;

/// Surface height deviation around [`BASE_Y`], in voxels.
///
/// 24 gives a band of 40..88, which matches `HeightMap::new`'s original base 64
/// amplitude 24 exactly. Every carver's tuning — `density::CARVE_THRESHOLD`, the
/// capsule and tube coverage bands, the sparse stamp radii — was calibrated
/// against that band, so keeping it is what stops this change from silently
/// re-tuning all four carvers.
pub const AMPLITUDE: f32 = 24.0;

/// Sea level, in absolute world Y.
///
/// Absolute, like the band: a window that does not reach it simply has no ocean.
pub const SEA_LEVEL: i32 = 62;

/// Surface height and sea level for a world.
///
/// Deliberately holds no cache: a column's height is one noise sample and a
/// multiply, cheaper than maintaining a map, and statelessness keeps parallel
/// generation trivially correct.
#[derive(Debug, Clone, Copy)]
pub struct HeightMap {
    seed: u32,
    sea_level: i32,
    base: f32,
    amplitude: f32,
}

impl HeightMap {
    /// The absolute band: base 64, amplitude 24, sea level 62.
    pub const fn absolute(seed: u32) -> Self {
        Self {
            seed,
            sea_level: SEA_LEVEL,
            base: BASE_Y,
            amplitude: AMPLITUDE,
        }
    }

    /// Base height 64, amplitude 24, with the sea level given.
    ///
    /// Kept because tests use it to isolate water behaviour from the band, and
    /// because the band it produces is the one [`Self::absolute`] now matches.
    pub const fn new(seed: u32, sea_level: i32) -> Self {
        Self {
            seed,
            sea_level,
            base: BASE_Y,
            amplitude: AMPLITUDE,
        }
    }

    /// Terrain for a world of any height.
    ///
    /// The height argument is accepted and ignored: the band is absolute, so the
    /// same seed gives the same terrain whether the window is 64 voxels tall or
    /// 512. Callers keep passing their world height so the clipping window stays
    /// visible at the call site, but it cannot change the answer — which is
    /// exactly the property `surface_height_ignores_world_size` asserts.
    pub const fn fitted_to_world(seed: u32, _world_height: u32) -> Self {
        Self::absolute(seed)
    }

    pub const fn base(&self) -> f32 {
        self.base
    }

    pub const fn amplitude(&self) -> f32 {
        self.amplitude
    }

    pub const fn with_sea_level(seed: u32, sea_level: i32) -> Self {
        Self::new(seed, sea_level)
    }

    pub const fn sea_level(&self) -> i32 {
        self.sea_level
    }

    /// Surface height at a world column, in absolute world voxel units.
    pub fn height(&self, wx: i32, wz: i32) -> i32 {
        // One generator per call is fine: `FastNoiseLite` construction is a
        // handful of field writes, and rebuilding it keeps this function pure
        // and therefore safe to call from a parallel iterator.
        let n = noise::surface(self.seed);
        let n2 = n.get_noise_2d(wx as f32, wz as f32);
        // A second, lower-frequency octave gives continents rather than a
        // uniform bump field, without a second generator.
        let mut c = noise::surface(self.seed ^ 0xA5A5_5A5A);
        c.set_frequency(Some(0.002));
        let continent = c.get_noise_2d(wx as f32, wz as f32);

        (self.base + continent * self.amplitude * 0.5 + n2 * self.amplitude * 0.5).round() as i32
    }

    /// Height clamped into a window `0..world_height`.
    ///
    /// Still a clamp rather than a rejection, because a caller asking for one
    /// column's height has nowhere to put "outside". Chunk generation does not
    /// use this: it clips by choosing which chunks to generate.
    pub fn height_clamped(&self, wx: i32, wz: i32, world_height: i32) -> i32 {
        self.height(wx, wz).clamp(0, world_height - 1)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn height_is_pure_and_seed_dependent() {
        let a = HeightMap::new(1337, 62);
        let b = HeightMap::new(1337, 62);
        let c = HeightMap::new(1338, 62);
        let mut differs = 0;
        for x in 0..64 {
            assert_eq!(a.height(x, 0), b.height(x, 0), "must be pure");
            if a.height(x, 0) != c.height(x, 0) {
                differs += 1;
            }
        }
        assert!(differs > 0, "different seeds must give different terrain");
    }

    #[test]
    fn height_stays_inside_the_configured_band() {
        let hm = HeightMap::new(1337, 62);
        let (lo, hi) = ((hm.base - hm.amplitude) as i32, (hm.base + hm.amplitude) as i32);
        for x in -512..512 {
            for z in -64..64 {
                let h = hm.height(x, z);
                assert!(h >= lo && h <= hi, "height {h} outside {lo}..={hi}");
            }
        }
    }

    #[test]
    fn neighbouring_columns_do_not_jump_wildly() {
        // A discontinuity here would show up as cliffs at chunk borders, which
        // is the classic symptom of a noise call missing its world offset.
        let hm = HeightMap::new(1337, 62);
        let mut max_jump = 0;
        for x in 0..512 {
            let d = (hm.height(x, 0) - hm.height(x + 1, 0)).abs();
            max_jump = max_jump.max(d);
        }
        // Amplitude 24 over frequency 0.0075 means a step of more than a few
        // voxels between adjacent columns is a bug, not terrain.
        assert!(max_jump <= 8, "adjacent columns differ by {max_jump}");
    }

    #[test]
    fn clamping_bounds_into_the_world_box() {
        let hm = HeightMap::new(1337, 62);
        for x in -256..256 {
            let h = hm.height_clamped(x, 0, 16);
            assert!((0..16).contains(&h), "clamped height {h} escaped 0..16");
        }
    }
}
