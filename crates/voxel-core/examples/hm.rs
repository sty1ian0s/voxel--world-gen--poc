fn main() {
    let hm = voxel_core::HeightMap::new(1337, 62);
    for x in [0i32, 8, 16, 32] {
        println!("height({x},0) = {}", hm.height(x, 0));
    }
    println!("sea_level = {}", hm.sea_level());
}
