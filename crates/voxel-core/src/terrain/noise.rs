//! Noise sources.
//!
//! `fastnoise-lite` provides the gradient noise; this module is the thin layer
//! that turns a world seed into reproducible generators and keeps the sampling
//! call sites identical between the two carvers, so the benchmark compares
//! carving strategies rather than noise setups.

use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

/// 2D surface noise, sampled once per world column.
pub fn surface(seed: u32) -> FastNoiseLite {
    let mut n = FastNoiseLite::new();
    n.set_seed(Some(seed as i32));
    n.set_noise_type(Some(NoiseType::OpenSimplex2));
    n.set_frequency(Some(0.0075));
    n.set_fractal_type(Some(FractalType::FBm));
    n.set_fractal_octaves(Some(4));
    n.set_fractal_lacunarity(Some(2.0));
    n.set_fractal_gain(Some(0.5));
    n
}

/// 3D cave density noise used by Pass 2A.
pub fn density(seed: u32) -> FastNoiseLite {
    let mut n = FastNoiseLite::new();
    // Offset the seed so the cave field is independent of the surface field;
    // sharing a seed makes caves visibly follow the hills.
    n.set_seed(Some(seed.wrapping_add(0x9E37_79B9) as i32));
    n.set_noise_type(Some(NoiseType::OpenSimplex2));
    n.set_frequency(Some(0.03));
    n.set_fractal_type(Some(FractalType::FBm));
    n.set_fractal_octaves(Some(3));
    n.set_fractal_lacunarity(Some(2.0));
    n.set_fractal_gain(Some(0.5));
    n
}

/// 3D cave field for the Minecraft-style carver in [`super::mc`].
///
/// The seed offset is `density`'s constant scaled by the field id, so each
/// field is uncorrelated with the others *and* with the surface and density
/// fields, by the same idiom rather than by four hand-picked numbers that
/// nothing keeps apart.
///
/// `field` picks which of the carver's fields this is (see `mc::Field`);
/// returning the generator rather than sampling it here is what lets the
/// carver control its own frequency and octave count per field.
pub fn cave(seed: u32, field: u32) -> FastNoiseLite {
    let mut n = FastNoiseLite::new();
    let id = field.wrapping_add(1);
    let offset = 0x9E37_79B9u32.wrapping_mul(id) ^ 0x85EB_CA6Bu32.wrapping_mul(id.rotate_left(13));
    n.set_seed(Some(seed.wrapping_add(offset) as i32));
    n.set_noise_type(Some(NoiseType::OpenSimplex2));
    n.set_fractal_type(Some(FractalType::FBm));
    n.set_fractal_lacunarity(Some(2.0));
    n.set_fractal_gain(Some(0.5));
    n
}

/// Deterministic per-region RNG for Pass 2B.
///
/// Uses SplitMix64 rather than a general-purpose RNG because the sparse carver
/// must be able to answer "what cave origins exist near this chunk" without
/// having generated the region first, and without allocating a table. A
/// counter-based hash is exactly that property: any region can be sampled
/// independently, in any order, on any thread, and get identical results.
#[derive(Debug, Clone, Copy)]
pub struct RegionRng {
    state: u64,
}

impl RegionRng {
    #[inline]
    pub const fn new(seed: u32, rx: i32, ry: i32, rz: i32) -> Self {
        // Fold the region coordinates into the seed with large odd constants so
        // adjacent regions do not produce correlated streams.
        let mut s = seed as u64;
        s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        s ^= (rx as i64 as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        s ^= (ry as i64 as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
        s ^= (rz as i64 as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
        Self {
            // A zero state is a fixed point of SplitMix64; force it non-zero.
            state: s | 1,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Top 24 bits give exactly the f32 mantissa; no rounding surprises.
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    #[inline]
    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cave_fields_are_independent_and_seed_dependent() {
        // The four fields the MC-style carver runs must not be the same field
        // sampled four times, or its three cave types would carve identically.
        let fields: Vec<_> = (0..4).map(|f| cave(1337, f)).collect();
        let mut identical_pairs = 0;
        for (i, a) in fields.iter().enumerate() {
            for (j, b) in fields.iter().enumerate().skip(i + 1) {
                if (1..24).all(|k| {
                    let p = k as f32 * 0.37;
                    a.get_noise_3d(p, p, p) == b.get_noise_3d(p, p, p)
                }) {
                    identical_pairs += 1;
                    eprintln!("fields {i} and {j} are identical");
                }
            }
        }
        assert_eq!(identical_pairs, 0, "cave fields are not independent");

        // And a different world seed must move every one of them.
        for f in 0..4 {
            let a = cave(1337, f);
            let c = cave(1338, f);
            assert!(
                (1..24).any(|k| {
                    let p = k as f32 * 0.37;
                    a.get_noise_3d(p, p, p) != c.get_noise_3d(p, p, p)
                }),
                "field {f} ignores the world seed"
            );
        }
    }

    #[test]
    fn cave_fields_are_reproducible() {
        let a = cave(1337, 2);
        let b = cave(1337, 2);
        for k in 0..64 {
            let p = k as f32 * 0.51;
            assert_eq!(
                a.get_noise_3d(p, p * 1.7, p * 0.3),
                b.get_noise_3d(p, p * 1.7, p * 0.3)
            );
        }
    }

    #[test]
    fn region_rng_is_reproducible_and_region_dependent() {
        let a = RegionRng::new(1337, 1, 2, 3);
        let b = RegionRng::new(1337, 1, 2, 3);
        let mut a = a;
        let mut b = b;
        for i in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64(), "diverged at draw {i}");
        }

        let mut c = RegionRng::new(1337, 1, 2, 4);
        let mut d = RegionRng::new(1337, 1, 2, 3);
        // Neighbouring regions must not produce the same stream, or caves
        // repeat on a grid.
        assert_ne!(c.next_u64(), d.next_u64());
    }

    #[test]
    fn region_rng_is_order_independent() {
        // The sparse carver samples regions in arbitrary order across threads,
        // so a region's stream must not depend on how many were drawn before.
        let mut forward = RegionRng::new(99, 0, 0, 0);
        let first = forward.next_u64();
        let mut after_others = RegionRng::new(99, 0, 0, 0);
        let _ = after_others.next_u64();
        let still_first = RegionRng::new(99, 0, 0, 0).next_u64();
        assert_eq!(first, still_first);
    }

    #[test]
    fn floats_stay_in_unit_interval() {
        let mut rng = RegionRng::new(5, 0, 0, 0);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
        let mut rng = RegionRng::new(5, 0, 0, 0);
        for _ in 0..10_000 {
            let v = rng.range_f32(-3.0, 7.0);
            assert!((-3.0..7.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn noise_is_deterministic_for_a_seed() {
        let a = surface(1337);
        let b = surface(1337);
        for x in 0..32 {
            assert_eq!(
                a.get_noise_2d(x as f32, 0.0),
                b.get_noise_2d(x as f32, 0.0),
                "same seed must give the same field"
            );
        }
        let c = surface(1338);
        // Sampled off the lattice origin: gradient noise is exactly 0.0 at
        // (0,0) for every seed, so comparing there proves nothing.
        let mut differs = false;
        for i in 1..32 {
            let p = (i as f32) * 0.37;
            if a.get_noise_2d(p, p) != c.get_noise_2d(p, p) {
                differs = true;
                break;
            }
        }
        assert!(differs, "different seeds must give different fields");
    }

    #[test]
    fn cave_and_surface_fields_are_independent() {
        // Sharing a seed would make caves track the surface; the offset in
        // `density` exists to prevent exactly that.
        let s = surface(1337);
        let d = density(1337);
        let mut same = 0;
        for i in 0..256 {
            let x = i as f32;
            if (s.get_noise_2d(x, x) - d.get_noise_3d(x, x, x)).abs() < 1e-6 {
                same += 1;
            }
        }
        assert!(same < 32, "fields appear correlated: {same}/256 identical");
    }
}
