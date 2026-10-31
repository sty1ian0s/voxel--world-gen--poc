//! Renders horizontal and vertical slices of the tube carver, to look at it.
use voxel_core::debug::ascii;
use voxel_core::terrain::{heightmap::HeightMap, tube};
use voxel_core::world::ChunkGrid;

fn main() {
    let side = 3i32;
    let hm = HeightMap::fitted_to_world(1337, 64);
    let mut grid = ChunkGrid::new(side as u32, 4, side as u32);
    for pos in grid.positions().collect::<Vec<_>>() {
        let c = grid.get_mut(pos);
        voxel_core::terrain::generate_surface(c, pos, &hm);
        tube::carve(c, 1337);
    }
    // Horizontal slices at three heights through the cave band.
    for y in [20i32, 26, 32] {
        println!("=== horizontal slice y={y} (air='.', stone='#') ===");
        print!("{}", ascii::carve_slice(&grid, y).to_string_with_header());
    }
    // Vertical slice: this is the one that shows a tube's profile.
    println!("=== vertical slice along z (x from 0..47, y from 12..44) ===");
    print!("{}", ascii::blocks_slice_xz(&grid, 24).to_string_with_header());
}
