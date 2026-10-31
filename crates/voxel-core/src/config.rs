//! Layered environment configuration.
//!
//! Precedence, lowest to highest: `.env` -> `.env.prod` (only when
//! `APP_ENV=prod` is already set) -> `.env.local` -> real process environment.
//! Loading happens once in [`Config::load`]; everything downstream reads the
//! parsed struct, so no hot path ever touches a string again.

use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Env {
    Dev,
    Prod,
}

impl Env {
    /// Dev is the default: prod is opt-in so an accidental release-mode run
    /// cannot silently disable validation.
    pub const fn is_dev(self) -> bool {
        matches!(self, Self::Dev)
    }

    pub const fn is_prod(self) -> bool {
        matches!(self, Self::Prod)
    }
}

impl fmt::Display for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Dev => "dev",
            Self::Prod => "prod",
        })
    }
}

impl FromStr for Env {
    type Err = ConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "dev" | "development" | "local" => Ok(Self::Dev),
            "prod" | "production" | "release" => Ok(Self::Prod),
            other => Err(ConfigError::BadValue {
                key: "APP_ENV",
                value: other.to_owned(),
                expected: "dev | prod",
            }),
        }
    }
}

/// Which cave carver Pass 2 runs.
///
/// All five carvers in the project, selectable by name through `VOXEL_CARVER`
/// and switchable at runtime. `Density` and `Sparse` are the two `PROJECT.md`
/// §4.2 names; `Mc`, `Capsule` and `Tube` were previously reachable only by
/// calling `mc::carve` / `capsule::carve` / `tube::carve` directly, which meant
/// `cavestat` could measure them but the app could not run them.
///
/// `Density` stays the default: it is what `.env` ships and what the benchmark
/// baseline assumes. `Tube` is the fastest of the three newer ones and is *not*
/// the default — it is opt-in by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Carver {
    /// Pass 2A: full 3D density field, `O(N^3)` noise samples.
    Density,
    /// Pass 2B: deterministic origin lookup + bounded SDF stamp carving, `O(S)`.
    Sparse,
    /// Minecraft-1.18-style noise caves: cheese, spaghetti and noodle fields.
    Mc,
    /// Pass 2C: capsule chains placed by 2D fields, Y by arithmetic.
    Capsule,
    /// Multi-node 2D interval tubes: all three MC cave types, no walk.
    Tube,
}

impl Carver {
    /// Every carver, in the order the benchmark and `cavestat` report them.
    ///
    /// A flat array rather than an iterator over a range, so adding a variant is
    /// a compile error at every `match` instead of a silently shorter table.
    pub const ALL: [Self; 5] = [
        Self::Density,
        Self::Sparse,
        Self::Mc,
        Self::Capsule,
        Self::Tube,
    ];

    /// The canonical `.env` spelling.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Density => "density",
            Self::Sparse => "sparse",
            Self::Mc => "mc",
            Self::Capsule => "capsule",
            Self::Tube => "tube",
        }
    }
}

impl fmt::Display for Carver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Carver {
    type Err = ConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "density" | "2a" | "on3" => Ok(Self::Density),
            "sparse" | "2b" | "stamp" => Ok(Self::Sparse),
            "mc" | "noise" | "1.18" => Ok(Self::Mc),
            "capsule" | "2c" => Ok(Self::Capsule),
            "tube" | "tubes" => Ok(Self::Tube),
            other => Err(ConfigError::BadValue {
                key: "VOXEL_CARVER",
                value: other.to_owned(),
                expected: "density | sparse | mc | capsule | tube",
            }),
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    BadValue {
        key: &'static str,
        value: String,
        expected: &'static str,
    },
    /// A value that parsed but is nonsense (zero-size world, inverted range).
    OutOfRange {
        key: &'static str,
        value: i64,
        expected: &'static str,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadValue {
                key,
                value,
                expected,
            } => write!(f, "{key}: {value:?} is invalid, expected {expected}"),
            Self::OutOfRange {
                key,
                value,
                expected,
            } => write!(f, "{key}: {value} is out of range, expected {expected}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Validated world extent in chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorldSize {
    pub const fn chunk_count(&self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }

    pub const fn voxel_count(&self) -> u64 {
        self.chunk_count() * 4096
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaveParams {
    /// Expected cave origins per 1000 voxels-cubed. Pass 2B iterates only the
    /// origins whose padded AABB can touch the requested chunk, so this number
    /// (not the world volume) is what its cost scales with.
    ///
    /// Per-1000 rather than per-voxel so the values stay readable. The default
    /// of 2.0 carves about 20% of the rock, deliberately matching the density
    /// carver's threshold so both approaches produce cave systems of a
    /// comparable size and the benchmark compares like with like.
    pub origin_density: f32,
    pub min_radius: f32,
    pub max_radius: f32,
}

/// Measures how much rock each carver removes, so the two can be compared.
///
/// Lives here rather than in the benchmark because the HUD and the parity test
/// both report it, and three copies of the same measurement would drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CarveParity {
    /// Solid voxels before carving, over the sampled band.
    pub solid: u64,
    /// Voxels the density carver turned to air.
    pub density_carved: u64,
    /// Voxels the sparse carver turned to air.
    pub sparse_carved: u64,
    /// Voxels the density carver sampled, for the `O(N^3)` comparison.
    pub density_samples: u64,
    /// Voxels the sparse carver evaluated an SDF against, for the `O(S)` one.
    pub sparse_voxels_tested: u64,
}

impl CarveParity {
    /// Samples both carvers over the same chunk band.
    ///
    /// `side` is the band width in chunks; the band spans two chunks vertically
    /// so the measurement covers underground rock rather than one slice.
    pub fn measure(seed: u32, params: &CaveParams, side: i32) -> Self {
        use crate::domain::{CHUNK_VOLUME, ChunkPos, VoxelIndex};
        use crate::terrain::{self, density, heightmap::HeightMap, sparse};
        use crate::world::Chunk;

        let hm = HeightMap::fitted_to_world(seed, 64);
        let mut out = Self::default();

        let air = |c: &Chunk| -> u64 {
            (0..CHUNK_VOLUME as u16)
                .filter(|&i| {
                    c.block(unsafe { VoxelIndex::from_raw_unchecked(i) })
                        .is_air()
                })
                .count() as u64
        };

        for cy in 1..3 {
            for cx in 0..side {
                for cz in 0..side {
                    let pos = ChunkPos::new(cx, cy, cz);

                    let mut base = Chunk::at(pos);
                    terrain::generate_surface(&mut base, pos, &hm);
                    let before = air(&base);
                    out.solid += CHUNK_VOLUME as u64 - before;

                    let mut d = base.clone();
                    out.density_samples += density::carve(&mut d, seed);
                    out.density_carved += air(&d) - before;

                    let mut s = base;
                    let stats = sparse::carve(&mut s, seed, params);
                    out.sparse_voxels_tested += stats.voxels_tested;
                    out.sparse_carved += air(&s) - before;
                }
            }
        }

        out
    }

    fn percent(carved: u64, solid: u64) -> f64 {
        100.0 * carved as f64 / solid.max(1) as f64
    }

    pub fn density_percent(&self) -> f64 {
        Self::percent(self.density_carved, self.solid)
    }

    pub fn sparse_percent(&self) -> f64 {
        Self::percent(self.sparse_carved, self.solid)
    }

    /// Difference in coverage, in percentage points.
    pub fn divergence(&self) -> f64 {
        (self.density_percent() - self.sparse_percent()).abs()
    }

    /// One-line summary for the log or the HUD.
    pub fn summary(&self) -> String {
        format!(
            "carve: density {:.1}% ({} samples) vs sparse {:.1}% ({} tested), divergence {:.1}pp",
            self.density_percent(),
            self.density_samples,
            self.sparse_percent(),
            self.sparse_voxels_tested,
            self.divergence()
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderConfig {
    pub force_software_gpu: bool,
    pub vsync: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub env: Env,
    pub world: WorldSize,
    pub seed: u32,
    pub carver: Carver,
    pub cave: CaveParams,
    pub validation: bool,
    pub debug_views: bool,
    pub frame_dump: bool,
    pub render: RenderConfig,
}

impl Config {
    /// Loads layered env files, then the process environment overrides them.
    pub fn load() -> Result<Self, ConfigError> {
        // `APP_ENV` is read before dotenv loading so the right overlay can be
        // picked; the file value is used only when the process did not set it.
        let env = match std::env::var("APP_ENV") {
            Ok(raw) => Env::from_str(&raw)?,
            Err(_) => Env::Dev,
        };

        // Missing .env files are fine (CI, packaged binaries); a malformed one
        // is not, but dotenvy only reports syntax problems, which we ignore
        // rather than refusing to boot on a stray comment.
        //
        // Two constraints pull against each other here:
        //   * `.env.prod` must be able to override `.env`, which plain
        //     `from_filename` will not do because it refuses to touch an
        //     already-set variable.
        //   * the real process environment must still beat every file, so
        //     `VOXEL_WORLD_X=64 cargo run` behaves as typed.
        //
        // So the real environment is snapshotted first, the files are applied
        // with `_override` (which lets later files win), and the snapshot is
        // then written back on top. That gets both orderings right, which
        // neither plain nor `_override` loading does on its own.
        let process_env: Vec<(String, String)> = std::env::vars()
            .filter(|(k, _)| k.starts_with("VOXEL_") || k == "APP_ENV" || k == "RUST_LOG")
            .collect();

        let _ = dotenvy::from_filename_override(".env");
        if env.is_prod() {
            let _ = dotenvy::from_filename_override(".env.prod");
        }
        let _ = dotenvy::from_filename_override(".env.local");

        for (k, v) in &process_env {
            // SAFETY: config is loaded once at startup, before any threads that
            // read the environment exist.
            unsafe { std::env::set_var(k, v) };
        }

        // `APP_ENV` is re-read so a shell override still selects the overlay it
        // was asked for, even though the files have now been applied.
        let env = raw("APP_ENV").map_or(Ok(env), |v| Env::from_str(&v))?;

        let world = WorldSize {
            x: num("VOXEL_WORLD_X", 8)?,
            y: num("VOXEL_WORLD_Y", 4)?,
            z: num("VOXEL_WORLD_Z", 8)?,
        };
        if world.x == 0 || world.y == 0 || world.z == 0 {
            return Err(ConfigError::OutOfRange {
                key: "VOXEL_WORLD_{X,Y,Z}",
                value: 0,
                expected: "> 0",
            });
        }

        let cave = CaveParams {
            origin_density: fnum("VOXEL_CAVE_ORIGIN_DENSITY", 2.0)?,
            min_radius: fnum("VOXEL_CAVE_MIN_RADIUS", 1.5)?,
            max_radius: fnum("VOXEL_CAVE_MAX_RADIUS", 4.0)?,
        };
        if cave.origin_density < 0.0 {
            return Err(ConfigError::OutOfRange {
                key: "VOXEL_CAVE_ORIGIN_DENSITY",
                value: cave.origin_density as i64,
                expected: ">= 0",
            });
        }
        if cave.min_radius <= 0.0 || cave.max_radius < cave.min_radius {
            return Err(ConfigError::OutOfRange {
                key: "VOXEL_CAVE_{MIN,MAX}_RADIUS",
                value: cave.max_radius as i64,
                expected: "0 < min <= max",
            });
        }

        let cfg = Self {
            env,
            world,
            seed: num("VOXEL_WORLD_SEED", 1337)?,
            carver: raw("VOXEL_CARVER").map_or(Ok(Carver::Density), |v| Carver::from_str(&v))?,
            cave,
            validation: flag("VOXEL_VALIDATION", true),
            debug_views: flag("VOXEL_DEBUG_VIEWS", true),
            frame_dump: flag("VOXEL_FRAME_DUMP", false),
            render: RenderConfig {
                force_software_gpu: flag("VOXEL_FORCE_SOFTWARE_GPU", false),
                vsync: flag("VOXEL_VSYNC", false),
            },
        };

        // Prod is the fast path; validation there would defeat the benchmark.
        let cfg = if cfg.env.is_prod() {
            Self {
                validation: false,
                ..cfg
            }
        } else {
            cfg
        };

        Ok(cfg)
    }

    /// World height in voxels.
    ///
    /// The streamed world is unbounded horizontally but fixed vertically, and
    /// the height is what the surface band is fitted to.
    pub fn world_height(&self) -> u32 {
        self.world.y * 16
    }

    /// Every knob that differs between a dev and a prod run, for the boot log.
    pub fn summary(&self) -> String {
        format!(
            "env={} world={}x{}x{} ({} chunks) seed={} carver={} cave[rho={} r={}..{}] \
             validation={} debug_views={} software_gpu={} vsync={}",
            self.env,
            self.world.x,
            self.world.y,
            self.world.z,
            self.world.chunk_count(),
            self.seed,
            self.carver,
            self.cave.origin_density,
            self.cave.min_radius,
            self.cave.max_radius,
            self.validation,
            self.debug_views,
            self.render.force_software_gpu,
            self.render.vsync,
        )
    }
}

fn raw(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn num<T: FromStr>(key: &str, default: T) -> Result<T, ConfigError> {
    match raw(key) {
        None => Ok(default),
        Some(v) => v.trim().parse().map_err(|_| ConfigError::BadValue {
            key: leak(key),
            value: v,
            expected: "a number",
        }),
    }
}

fn fnum(key: &str, default: f32) -> Result<f32, ConfigError> {
    num(key, default)
}

/// Accepts `1/0`, `true/false`, `yes/no`, `on/off`. Anything else is a typo and
/// is treated as unset rather than silently flipping a debug switch.
fn flag(key: &str, default: bool) -> bool {
    match raw(key) {
        None => default,
        Some(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => {
                log::warn!("{key}={other:?} is not a boolean, using {default}");
                default
            }
        },
    }
}

/// `ConfigError` wants `&'static str` keys; the keys are compile-time literals
/// at every call site, so this only launders the lifetime of `key` above.
fn leak(key: &str) -> &'static str {
    Box::leak(key.to_owned().into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_is_dev_by_default_and_prod_is_explicit() {
        assert!(Env::Dev.is_dev() && !Env::Dev.is_prod());
        assert!(Env::Prod.is_prod());
        assert_eq!("prod".parse::<Env>().unwrap(), Env::Prod);
        assert_eq!("PRODUCTION".parse::<Env>().unwrap(), Env::Prod);
        assert!("banana".parse::<Env>().is_err());
    }

    #[test]
    fn carver_parses_both_spellings() {
        assert_eq!("density".parse::<Carver>().unwrap(), Carver::Density);
        assert_eq!("2b".parse::<Carver>().unwrap(), Carver::Sparse);
        assert!("melt".parse::<Carver>().is_err());
    }

    #[test]
    fn flag_rejects_typos_without_flipping() {
        // SAFETY: single-threaded test process; each key is unique to this test.
        unsafe { std::env::set_var("VOXEL_TEST_FLAG", "maybe") };
        assert!(flag("VOXEL_TEST_FLAG", true));
        unsafe { std::env::set_var("VOXEL_TEST_FLAG", "off") };
        assert!(!flag("VOXEL_TEST_FLAG", true));
        unsafe { std::env::remove_var("VOXEL_TEST_FLAG") };
    }

    #[test]
    fn num_reports_the_offending_key() {
        unsafe { std::env::set_var("VOXEL_TEST_NUM", "banana") };
        let err = num("VOXEL_TEST_NUM", 3u32).unwrap_err();
        assert!(err.to_string().contains("VOXEL_TEST_NUM"), "{err}");
        unsafe { std::env::remove_var("VOXEL_TEST_NUM") };
    }
}
