//! Image cross-sections, without an image dependency.
//!
//! Writes binary PGM (grayscale) and PPM (colour) — formats that need a
//! two-line header and raw bytes, which is less code than pulling in an encoder
//! and is readable by every image viewer. Used to eyeball large worlds where a
//! 256x256 ASCII dump is unreadable.
//!
//! Colour mapping is fixed and documented so two dumps from different runs are
//! directly comparable, which is what makes them useful for regression checks.

use super::Axis;
use crate::domain::BlockId;
use crate::world::ChunkGrid;
use std::io::{self, Write};
use std::path::Path;

/// RGB for a block, chosen so terrain reads naturally.
pub const fn block_rgb(block: BlockId) -> [u8; 3] {
    match block {
        BlockId::AIR => [135, 206, 235], // sky
        BlockId::STONE => [110, 110, 118],
        BlockId::DIRT => [134, 96, 67],
        BlockId::GRASS => [86, 155, 62],
        BlockId::SAND => [222, 205, 135],
        BlockId::WATER => [50, 90, 200],
        // Magenta for anything unrecognised: impossible to miss in a dump.
        _ => [255, 0, 255],
    }
}

/// Renders a horizontal world slice in colour and writes it as a PPM.
pub fn write_blocks_slice_ppm(
    path: impl AsRef<Path>,
    grid: &ChunkGrid,
    level: i32,
) -> io::Result<()> {
    let (sx, _, sz) = grid.size();
    let width = sx * 16;
    let height = sz * 16;

    let mut buf = Vec::with_capacity((width * height * 3) as usize);
    for wz in 0..height as i32 {
        for wx in 0..width as i32 {
            buf.extend_from_slice(&block_rgb(super::ascii::voxel_at(grid, wx, level, wz)));
        }
    }
    write_ppm(path, width, height, &buf)
}

/// Renders sunlight as grayscale and writes it as a PGM.
///
/// Values are scaled to `0..=255` so a 4-bit light level still spans the full
/// output range and gradients are visible.
pub fn write_sunlight_slice_pgm(
    path: impl AsRef<Path>,
    grid: &ChunkGrid,
    level: i32,
) -> io::Result<()> {
    let (sx, _, sz) = grid.size();
    let width = sx * 16;
    let height = sz * 16;

    let mut buf = Vec::with_capacity((width * height) as usize);
    for wz in 0..height as i32 {
        for wx in 0..width as i32 {
            let light = super::ascii::sunlight_at(grid, wx, level, wz).raw();
            buf.push((light as u16 * 255 / 15) as u8);
        }
    }
    write_pgm(path, width, height, &buf)
}

/// Writes a vertical slice (constant Z) as a PPM, highest Y at the top.
pub fn write_cross_section_ppm(
    path: impl AsRef<Path>,
    grid: &ChunkGrid,
    level_z: i32,
) -> io::Result<()> {
    let (sx, sy, _) = grid.size();
    let width = sx * 16;
    let height = sy * 16;

    let mut buf = Vec::with_capacity((width * height * 3) as usize);
    for wy in (0..height as i32).rev() {
        for wx in 0..width as i32 {
            buf.extend_from_slice(&block_rgb(super::ascii::voxel_at(grid, wx, wy, level_z)));
        }
    }
    write_ppm(path, width, height, &buf)
}

fn write_ppm(path: impl AsRef<Path>, w: u32, h: u32, rgb: &[u8]) -> io::Result<()> {
    debug_assert_eq!(rgb.len(), (w * h * 3) as usize);
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "P6\n{w} {h}\n255\n")?;
    f.write_all(rgb)?;
    f.flush()
}

fn write_pgm(path: impl AsRef<Path>, w: u32, h: u32, gray: &[u8]) -> io::Result<()> {
    debug_assert_eq!(gray.len(), (w * h) as usize);
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "P5\n{w} {h}\n255\n")?;
    f.write_all(gray)?;
    f.flush()
}

/// The slice orientations a dump can use, for the CLI.
pub const ALL_AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ChunkPos;

    fn world() -> ChunkGrid {
        let mut grid = ChunkGrid::new(2, 1, 2);
        for pos in [ChunkPos::new(0, 0, 0), ChunkPos::new(1, 0, 1)] {
            let c = grid.get_mut(pos);
            c.fill(BlockId::GRASS);
        }
        grid
    }

    #[test]
    fn ppm_header_and_size_are_correct() {
        let dir = std::env::temp_dir().join("voxel-debug-test-ppm");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("slice.ppm");
        write_blocks_slice_ppm(&path, &world(), 0).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let header = b"P6\n32 32\n255\n";
        assert!(bytes.starts_with(header), "bad header");
        // 32x32 RGB plus the header.
        assert_eq!(bytes.len(), header.len() + 32 * 32 * 3);
        assert_eq!(bytes[header.len()..header.len() + 3], block_rgb(BlockId::GRASS));
    }

    #[test]
    fn pgm_scales_light_to_the_full_range() {
        let dir = std::env::temp_dir().join("voxel-debug-test-pgm");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("light.pgm");
        write_sunlight_slice_pgm(&path, &world(), 0).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let header = b"P5\n32 32\n255\n";
        assert!(bytes.starts_with(header));
        assert_eq!(bytes.len(), header.len() + 32 * 32);
        // This world was filled without lighting, so everything is dark.
        assert!(bytes[header.len()..].iter().all(|&b| b == 0));
    }

    #[test]
    fn cross_section_is_flipped_so_sky_is_on_top() {
        let dir = std::env::temp_dir().join("voxel-debug-test-cross");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cross.ppm");

        let mut grid = ChunkGrid::new(1, 2, 1);
        // Bottom chunk solid, top chunk air: sky must render at the top.
        grid.get_mut(ChunkPos::new(0, 0, 0)).fill(BlockId::STONE);
        write_cross_section_ppm(&path, &grid, 0).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let header = b"P6\n16 32\n255\n";
        assert!(bytes.starts_with(header));
        let px = |x: usize, y: usize| {
            let i = header.len() + (y * 16 + x) * 3;
            [bytes[i], bytes[i + 1], bytes[i + 2]]
        };
        assert_eq!(px(0, 0), block_rgb(BlockId::AIR), "top row must be sky");
        assert_eq!(px(0, 31), block_rgb(BlockId::STONE), "bottom row is stone");
    }

    #[test]
    fn unknown_blocks_are_not_silently_sky() {
        assert_eq!(block_rgb(BlockId::new(250)), [255, 0, 255]);
    }
}
