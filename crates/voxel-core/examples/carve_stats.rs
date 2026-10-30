//! Prints measured carver cost. Temporary probe; replaced by benches in Task 11.
use std::time::Instant;
use voxel_core::config::CaveParams;
use voxel_core::domain::ChunkPos;
use voxel_core::terrain::{density, heightmap::HeightMap, sparse};
use voxel_core::world::Chunk;

fn main() {
    let seed = 1337u32;
    let hm = HeightMap::new(seed, 0);
    let params = CaveParams { origin_density: 0.02, min_radius: 1.5, max_radius: 4.0 };

    for &side in &[4i32, 8, 16] {
        let n = (side * side) as u64;

        // Pre-generate surfaces so the timing measures carving only, not the
        // shared Pass 1 that both strategies pay for.
        let mut base = Vec::new();
        for z in 0..side {
            for x in 0..side {
                let pos = ChunkPos::new(x, 0, z);
                let mut c = Chunk::at(pos);
                voxel_core::terrain::generate_surface(&mut c, pos, &hm);
                base.push(c);
            }
        }

        let t = Instant::now();
        let mut samples = 0u64;
        for c in &mut base {
            samples += density::carve(c, seed);
        }
        let d_ms = t.elapsed().as_secs_f64() * 1000.0;

        for c in &mut base {
            let pos = c.pos();
            voxel_core::terrain::generate_surface(c, pos, &hm);
        }

        let t = Instant::now();
        let mut tested = 0u64;
        let mut regions = 0u64;
        for c in &mut base {
            let s = sparse::carve(c, seed, &params);
            tested += s.voxels_tested;
            regions += s.regions_visited;
        }
        let s_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!(
            "{side}x{side} chunks ({n}): density {samples} samples / {d_ms:.1}ms | \
             sparse {tested} tested / {regions} regions / {s_ms:.1}ms | speedup {:.1}x",
            d_ms / s_ms.max(0.0001)
        );
    }
}
