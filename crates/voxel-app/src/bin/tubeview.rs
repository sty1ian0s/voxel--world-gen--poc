//! Renders horizontal and vertical slices of the tube carver, to look at it.
use voxel_core::debug::ascii;
use voxel_core::terrain::heightmap::HeightMap;
use voxel_core::terrain::{measured_band_chunks, tube};
use voxel_core::world::ChunkGrid;

fn main() {
    let side = 3i32;
    let band = measured_band_chunks();
    let hm = HeightMap::fitted_to_world(1337, 128);
    // Tall enough to contain the band the carver actually places caves in.
    let mut grid = ChunkGrid::new(side as u32, band.end as u32 + 1, side as u32);
    for pos in grid.positions().collect::<Vec<_>>() {
        let c = grid.get_mut(pos);
        voxel_core::terrain::generate_surface(c, pos, &hm);
        tube::carve(c, 1337);
    }
    // Horizontal slices through the cave band, derived from the band itself:
    // hardcoded heights went stale the moment cave placement became absolute,
    // and an empty render then reads as "the carver is broken".
    let lo = band.start * 16;
    let hi = band.end * 16;
    debug_assert!(lo < hi);
    for y in [lo + 12, lo + 22, (lo + hi) / 2, hi - 20] {
        println!("=== horizontal slice y={y} (air='.', stone='#') ===");
        print!("{}", ascii::carve_slice(&grid, y).to_string_with_header());
    }
    // Vertical slice: this is the one that shows a tube's profile.
    println!("=== vertical slice along z (x from 0..47) ===");
    print!("{}", ascii::blocks_slice_xz(&grid, 24).to_string_with_header());
}
