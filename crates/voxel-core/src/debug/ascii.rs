//! ASCII slice views.
//!
//! One character per voxel, so a chunk slice is 16x16 characters and a whole
//! world can be read without an image viewer. This is the fastest way to see
//! whether a carver produced connected caves, whether the surface is flat in
//! the wrong places, or whether chunks are offset from each other.

use super::Axis;
use crate::domain::{BlockId, ChunkPos, LightLevel, LocalPos, VoxelIndex, CHUNK_SIZE};
use crate::world::ChunkGrid;

/// Renders a block id as one character.
///
/// Distinct and visually weighted so a slice reads as terrain at a glance.
pub fn block_glyph(block: BlockId) -> char {
    match block {
        BlockId::AIR => ' ',
        BlockId::STONE => '#',
        BlockId::DIRT => ':',
        BlockId::GRASS => '"',
        BlockId::SAND => '.',
        BlockId::WATER => '~',
        // Anything unknown is surfaced loudly rather than rendered as air,
        // since a wrong id is exactly the kind of bug these views exist for.
        _ => '?',
    }
}

/// Renders a light level as a digit, `0-9` then `a-f` for `10-15`.
pub fn light_glyph(level: LightLevel) -> char {
    let v = level.raw();
    if v < 10 {
        (b'0' + v) as char
    } else {
        (b'a' + (v - 10)) as char
    }
}

/// A rendered slice, with a header naming exactly what it shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slice {
    pub axis: Axis,
    /// Coordinate of the slice along `axis`, in world voxels.
    pub level: i32,
    /// Rows top to bottom, each one row of the slice.
    pub rows: Vec<String>,
}

impl Slice {
    pub fn to_string_with_header(&self) -> String {
        let mut s = format!(
            "slice axis={} level={} ({}x{})\n",
            self.axis.name(),
            self.level,
            self.rows.first().map_or(0, |r| r.len()),
            self.rows.len()
        );
        for row in &self.rows {
            s.push_str(row);
            s.push('\n');
        }
        s
    }
}

/// Renders one horizontal slice of a chunk grid as blocks.
///
/// `level` is a world Y coordinate; only chunks containing that level are read,
/// so dumping a slice of a large world does not touch the rest of it.
pub fn blocks_slice(grid: &ChunkGrid, level: i32) -> Slice {
    let (sx, _, sz) = grid.size();
    let width = sx as usize * CHUNK_SIZE;
    let depth = sz as usize * CHUNK_SIZE;

    let mut rows = Vec::with_capacity(depth);
    for wz in 0..depth {
        let mut row = String::with_capacity(width);
        for wx in 0..width {
            row.push(block_glyph(voxel_at(grid, wx as i32, level, wz as i32)));
        }
        rows.push(row);
    }

    Slice {
        axis: Axis::Y,
        level,
        rows,
    }
}

/// Renders one vertical slice (constant Z) of a chunk grid as blocks, with the
/// highest Y at the top so it reads like a cross-section.
pub fn blocks_slice_xz(grid: &ChunkGrid, level_z: i32) -> Slice {
    let (sx, sy, _) = grid.size();
    let width = sx as usize * CHUNK_SIZE;
    let height = sy as usize * CHUNK_SIZE;

    let mut rows = Vec::with_capacity(height);
    for wy in (0..height as i32).rev() {
        let mut row = String::with_capacity(width);
        for wx in 0..width as i32 {
            row.push(block_glyph(voxel_at(grid, wx, wy, level_z)));
        }
        rows.push(row);
    }

    Slice {
        axis: Axis::Z,
        level: level_z,
        rows,
    }
}

/// Renders a horizontal slice showing sunlight instead of blocks.
pub fn sunlight_slice(grid: &ChunkGrid, level: i32) -> Slice {
    let (sx, _, sz) = grid.size();
    let width = sx as usize * CHUNK_SIZE;
    let depth = sz as usize * CHUNK_SIZE;

    let mut rows = Vec::with_capacity(depth);
    for wz in 0..depth {
        let mut row = String::with_capacity(width);
        for wx in 0..width {
            let light = sunlight_at(grid, wx as i32, level, wz as i32);
            row.push(light_glyph(light));
        }
        rows.push(row);
    }

    Slice {
        axis: Axis::Y,
        level,
        rows,
    }
}

/// Renders a slice distinguishing air, solid and carved-from-solid.
///
/// The `o` marking empty space that the carver created (air surrounded by
/// solid) is what makes cave connectivity checkable by eye.
pub fn carve_slice(grid: &ChunkGrid, level: i32) -> Slice {
    let (sx, _, sz) = grid.size();
    let width = sx as usize * CHUNK_SIZE;
    let depth = sz as usize * CHUNK_SIZE;

    let mut rows = Vec::with_capacity(depth);
    for wz in 0..depth {
        let mut row = String::with_capacity(width);
        for wx in 0..width {
            let x = wx as i32;
            let z = wz as i32;
            let b = voxel_at(grid, x, level, z);
            let glyph = if !b.is_air() {
                block_glyph(b)
            } else if enclosed_at(grid, x, level, z) {
                // Empty, but surrounded on all six sides by non-air: produced
                // by a carver rather than by the surface pass.
                'o'
            } else {
                ' '
            };
            row.push(glyph);
        }
        rows.push(row);
    }

    Slice {
        axis: Axis::Y,
        level,
        rows,
    }
}

/// Reads one voxel, returning air for anything outside the world so slices at
/// the boundary do not need special handling.
pub fn voxel_at(grid: &ChunkGrid, wx: i32, wy: i32, wz: i32) -> BlockId {
    with_voxel(grid, wx, wy, wz, BlockId::AIR, |c, i| c.block(i))
}

pub fn sunlight_at(grid: &ChunkGrid, wx: i32, wy: i32, wz: i32) -> LightLevel {
    with_voxel(grid, wx, wy, wz, LightLevel::DARK, |c, i| c.sun(i))
}

pub fn block_light_at(grid: &ChunkGrid, wx: i32, wy: i32, wz: i32) -> LightLevel {
    with_voxel(grid, wx, wy, wz, LightLevel::DARK, |c, i| c.block_light(i))
}

/// True if the voxel is air but has a non-air neighbour on all six sides.
fn enclosed_at(grid: &ChunkGrid, wx: i32, wy: i32, wz: i32) -> bool {
    const NEIGHBOURS: [(i32, i32, i32); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];
    NEIGHBOURS
        .iter()
        .all(|&(dx, dy, dz)| !voxel_at(grid, wx + dx, wy + dy, wz + dz).is_air())
}

/// Shared lookup: converts world coordinates to a chunk and index, returning
/// `default` for out-of-world coordinates.
fn with_voxel<T>(
    grid: &ChunkGrid,
    wx: i32,
    wy: i32,
    wz: i32,
    default: T,
    read: impl FnOnce(&crate::world::Chunk, VoxelIndex) -> T,
) -> T {
    let pos = ChunkPos::from_world(wx, wy, wz);
    if !grid.contains(pos) {
        return default;
    }
    let local = LocalPos::from_world(wx, wy, wz);
    read(grid.get(pos), VoxelIndex::from_local(local))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::heightmap::HeightMap;

    fn flat_world(fill: BlockId, size: (u32, u32, u32)) -> ChunkGrid {
        let mut grid = ChunkGrid::new(size.0, size.1, size.2);
        let positions: Vec<_> = grid.positions().collect();
        for pos in positions {
            let c = grid.get_mut(pos);
            c.fill(fill);
            c.refresh_flags();
        }
        grid
    }

    #[test]
    fn slice_dimensions_match_the_world() {
        let grid = flat_world(BlockId::STONE, (2, 1, 3));
        let s = blocks_slice(&grid, 0);
        assert_eq!(s.rows.len(), 3 * 16, "depth rows");
        assert!(s.rows.iter().all(|r| r.len() == 2 * 16), "width columns");
        assert_eq!(s.axis, Axis::Y);
    }

    #[test]
    fn glyphs_distinguish_every_known_block() {
        let glyphs: Vec<char> = [
            BlockId::AIR,
            BlockId::STONE,
            BlockId::DIRT,
            BlockId::GRASS,
            BlockId::SAND,
            BlockId::WATER,
        ]
        .iter()
        .map(|&b| block_glyph(b))
        .collect();
        let unique: std::collections::HashSet<char> = glyphs.iter().copied().collect();
        assert_eq!(unique.len(), glyphs.len(), "block glyphs must be distinct");
        assert_eq!(block_glyph(BlockId::new(200)), '?', "unknown ids are loud");
    }

    #[test]
    fn light_glyphs_cover_the_whole_range() {
        assert_eq!(light_glyph(LightLevel::DARK), '0');
        assert_eq!(light_glyph(LightLevel::new(9)), '9');
        assert_eq!(light_glyph(LightLevel::new(10)), 'a');
        assert_eq!(light_glyph(LightLevel::MAX), 'f');
    }

    #[test]
    fn surface_slice_shows_grass_above_stone() {
        // A vertical slice has the highest Y at the top, so grass must appear
        // above stone for a normal surface.
        let hm = HeightMap::new(1337, 0);
        let mut grid = ChunkGrid::new(1, 2, 1);
        let pos = ChunkPos::new(0, 0, 0);
        let mut c = crate::world::Chunk::at(pos);
        crate::terrain::generate_surface(&mut c, pos, &hm);
        *grid.get_mut(pos) = c;

        let s = blocks_slice_xz(&grid, 8);
        assert_eq!(s.axis, Axis::Z);
        assert_eq!(s.rows.len(), 32, "world is 2 chunks tall");

        // The first non-space column must be the surface grass.
        let col_x = 3;
        let first = s
            .rows
            .iter()
            .find_map(|r| r.chars().nth(col_x))
            .expect("a column");
        assert!(
            first == '"' || first == ' ',
            "topmost glyph should be grass or air, got {first:?}"
        );
    }

    #[test]
    fn out_of_world_reads_are_air_not_panics() {
        let grid = flat_world(BlockId::STONE, (1, 1, 1));
        assert!(voxel_at(&grid, -1, 0, 0).is_air());
        assert!(voxel_at(&grid, 0, -1, 0).is_air());
        assert!(voxel_at(&grid, 16, 0, 0).is_air());
        assert!(voxel_at(&grid, 0, 0, 16).is_air());
        assert_eq!(voxel_at(&grid, 0, 0, 0), BlockId::STONE);
        assert_eq!(sunlight_at(&grid, -5, -5, -5), LightLevel::DARK);
    }

    #[test]
    fn carve_slice_marks_enclosed_air_only() {
        let mut grid = flat_world(BlockId::STONE, (1, 1, 1));
        // A single air pocket deep inside the solid chunk.
        let pos = ChunkPos::ORIGIN;
        grid.get_mut(pos)
            .set_block(VoxelIndex::from_parts(5, 5, 5), BlockId::AIR);

        let s = carve_slice(&grid, 5);
        // Row index equals wz, column index equals wx for an x/z slice.
        let glyph = s.rows[5].chars().nth(5).unwrap();
        assert_eq!(glyph, 'o', "enclosed air must be marked as carved");

        // Air at the world edge has outside air neighbours, so it is not
        // "carved"; it is just empty sky.
        let sky = carve_slice(&grid, 100);
        assert!(
            sky.rows.iter().all(|r| r.chars().all(|c| c == ' ')),
            "empty sky must render as blank"
        );
    }

    #[test]
    fn header_names_the_view() {
        let grid = flat_world(BlockId::STONE, (1, 1, 1));
        let text = blocks_slice(&grid, 7).to_string_with_header();
        assert!(text.starts_with("slice axis=y level=7 (16x16)"), "got: {text}");
        assert_eq!(text.lines().count(), 1 + 16);
    }
}
