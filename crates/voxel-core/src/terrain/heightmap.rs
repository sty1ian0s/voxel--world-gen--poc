//! Pass 1: 2D surface heightmap.
//!
//! Sampled once per world column and shared by both Pass 2 strategies, so the
//! benchmark compares cave carving rather than terrain setup. Pure function of
//! `(seed, x, z)`, which is what lets chunk generation run in any order on any
//! thread and still agree.

use super::noise;

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
    /// Base height 64, amplitude 24, sea level 62 — values chosen so that the
    /// default config's shallow worlds still straddle water and land.
    pub const fn new(seed: u32, sea_level: i32) -> Self {
        Self {
            seed,
            sea_level,
            base: 64.0,
            amplitude: 24.0,
        }
    }

    pub const fn with_sea_level(seed: u32, sea_level: i32) -> Self {
        Self::new(seed, sea_level)
    }

    pub const fn sea_level(&self) -> i32 {
        self.sea_level
    }

    /// Surface height at a world column, in world voxel units.
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

    /// Height clamped into this POC's world box.
    ///
    /// shortcut: the world is a fixed-size box, so heights outside it are
    /// clamped flat. Upgrade to an infinite world when streaming lands.
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
