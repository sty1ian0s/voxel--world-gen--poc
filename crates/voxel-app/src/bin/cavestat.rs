//! `cavestat` — cave shape statistics for each carver, side by side.
//!
//! Coverage parity is not likeness: a field of disconnected blobs and a
//! connected tunnel system can remove identical fractions of rock and look
//! nothing alike. This binary prints the statistics that separate them, so a
//! carver change can be judged on shape rather than only on speed.
//!
//! ```text
//! cargo run --release -p voxel-app --bin cavestat
//! ```

use voxel_core::config::{CaveParams, Config};
use voxel_core::debug::caves::{self, CaveMask};
use voxel_core::debug::report;
use voxel_core::terrain::{capsule, mc, tube};
use voxel_core::{Carver, Env, RenderConfig, WorldSize};

/// Chunks per side. Big enough that a connected tunnel system can span several
/// chunks, which is the only way connectivity means anything.
const SIDE: u32 = 8;

const ALL: [Carver; 2] = [Carver::Density, Carver::Sparse];

fn main() {
    let params = CaveParams {
        origin_density: 2.0,
        min_radius: 1.5,
        max_radius: 4.0,
    };

    println!(
        "{SIDE}x{SIDE}x{SIDE} chunks, seed 1337, origin_density 2\n\
         connectivity = share of cave air in the single largest connected component\n\
         MC noise caves are one connected field, so a high score is the MC-like result\n"
    );
    println!(
        "{:<9} {:>7} {:>13} {:>11} {:>7} {:>9}",
        "carver", "void%", "connectivity", "components", "trunk", "gen_ms"
    );
    println!("{}", "-".repeat(62));

    // The two new carvers are not behind the `Carver` enum yet, so they are
    // measured directly: same grid, same statistics, same world size, so the
    // rows are comparable to the enum-driven ones below.
    /// A carver callable directly, for the two not yet behind `Carver`.
    type CarveFn = fn(&mut voxel_core::world::Chunk, u32) -> u64;
    let direct: [(&str, CarveFn); 3] = [
        ("mc", mc::carve),
        ("capsule", capsule::carve),
        ("tube", tube::carve),
    ];
    for (name, carve) in direct {
        let hm = voxel_core::terrain::heightmap::HeightMap::fitted_to_world(1337, SIDE * 16);
        let mut grid = voxel_core::world::ChunkGrid::new(SIDE, SIDE, SIDE);
        let start = std::time::Instant::now();
        let mut samples = 0u64;
        for pos in grid.positions().collect::<Vec<_>>() {
            let c = grid.get_mut(pos);
            voxel_core::terrain::generate_surface(c, pos, &hm);
            samples += carve(c, 1337);
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let stats = caves::measure(&grid, SIDE as i32, CaveMask::Underground);
        println!(
            "{:<9} {:>6.1}% {:>12.1}% {:>11} {:>7} {:>9.1}",
            name, stats.void_fraction() * 100.0,
            stats.largest_component_share() * 100.0,
            stats.components, stats.trunk_length, ms,
        );
        println!(
            "{:<9} samples: {samples} | depth/col {:.2} max_col {} mean_col {:.1} sv {:.3} | tunnels={}",
            "",
            stats.depth_per_column,
            stats.max_column_height,
            stats.mean_column_height,
            stats.surface_to_volume(),
            stats.looks_like_tunnels()
        );
    }

    for carver in ALL {
        let cfg = Config {
            env: Env::Dev,
            world: WorldSize {
                x: SIDE,
                y: SIDE,
                z: SIDE,
            },
            seed: 1337,
            carver,
            cave: params,
            validation: false,
            debug_views: false,
            frame_dump: false,
            render: RenderConfig {
                force_software_gpu: false,
                vsync: false,
            },
        };
        let (grid, rep) = report::generate_world(&cfg);
        let stats = caves::measure(&grid, SIDE as i32, CaveMask::Underground);
        println!(
            "{:<9} {:>6.1}% {:>12.1}% {:>11} {:>7} {:>9.1}",
            carver.to_string(),
            stats.void_fraction() * 100.0,
            stats.largest_component_share() * 100.0,
            stats.components,
            stats.trunk_length,
            rep.elapsed.as_secs_f64() * 1000.0,
        );
    }
}
