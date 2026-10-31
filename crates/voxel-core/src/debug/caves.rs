//! Cave-shape statistics: the yardstick that decides whether a carver produces
//! Minecraft-like caves.
//!
//! This module exists because the project previously had no way to judge cave
//! *quality*, only cave *quantity*. The parity test asserted that both carvers
//! remove a comparable fraction of rock, and coverage parity was reported as if
//! it settled likeness. It does not: a field of disconnected spheres and a
//! system of winding connected tunnels can remove exactly the same 20% of the
//! rock and look nothing alike.
//!
//! The statistics below are chosen to separate those two cases, and where
//! possible they are compared against documented Minecraft 1.18 numbers
//! (minecraft.wiki, "Cave"): main rooms 5-15 blocks in diameter, trunks 85-112
//! blocks long, branches 2-7 blocks thick, noodle tunnels 1-5 blocks wide. The
//! single most diagnostic one is [`CaveStats::largest_component_share`]:
//! Minecraft's noise caves are connected because they are the near-zero level
//! set of a continuous 3D field, so almost all their air sits in one giant
//! component. Independently stamped blobs do not, however many of them there
//! are.
//!
//! Measurement is over a 3D band of chunks, not one slice, because a single
//! slice cannot tell a tunnel crossing it from a sphere intersecting it.

use std::collections::VecDeque;

use crate::domain::{CHUNK_VOLUME, ChunkPos, VoxelIndex};
use crate::world::ChunkGrid;

/// How the cave voxels were selected from the world.
///
/// Sky and ocean are excluded from every count, because they are air that no
/// carver produced. Counting them would let a carver "win" on connectivity by
/// being adjacent to the surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaveMask {
    /// Air that is not open to the sky: the honest definition of a cave, and
    /// the one the rest of the project already uses for `carved_voxel_count`.
    Underground,
    /// Every air voxel, sky included. Useful for a sanity check, and never the
    /// number to quote as cave quality.
    AllAir,
}

/// Shape statistics for one world's caves.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CaveStats {
    /// Cave voxels, i.e. voxels matching the mask.
    pub cave_voxels: u64,
    /// Solid (non-air) voxels in the measured band, the denominator for
    /// [`Self::void_fraction`].
    pub solid_voxels: u64,
    /// Number of 6-connected cave components.
    pub components: u64,
    /// Voxels in the largest single component. Compare against
    /// [`Self::cave_voxels`] via [`Self::largest_component_share`].
    pub largest_component: u64,
    /// Components of at least 16 voxels: a blob or bigger. Fragments below this
    /// are speckle, and counting them as "caves" would flatter any carver.
    pub components_ge_16: u64,
    /// Mean cross-sectional area of the largest component per y slice, a proxy
    /// for tunnel width. Tunnels are thin in two axes; rooms are wide in both.
    pub mean_slice_area: f64,
    /// Longest extent of the largest component along any single axis.
    pub trunk_length: u64,
    /// Cave voxels that have at least one non-cave 6-neighbour: the surface.
    pub surface_voxels: u64,
    /// Mean filled `y`-extent per occupied `(x, z)` column of the largest
    /// component. A thin tube occupies a few voxels of a column; a sheet or a
    /// void fills many.
    pub mean_column_height: f64,
    /// The tallest `y`-extent of any single `(x, z)` column of the largest
    /// component. A tunnel is thin, so this stays near its width; a per-node
    /// box that gives every column the same interval fills far more.
    pub max_column_height: u64,
    /// Area of the largest component's `(x, z)` footprint, in columns.
    ///
    /// The metric that catches a thin *sheet*: such a cave has a small column
    /// height and looks innocent on every vertical measure, but its footprint is
    /// enormous. A cave system of tunnels covers area too, but it covers it in
    /// strips, so the ratio below separates them.
    pub footprint_columns: u64,
    /// Cave voxels per footprint column: how deep the cave is on average,
    /// measured over the columns it actually occupies.
    ///
    /// A tube is several voxels deep; a one-voxel sheet is exactly 1.
    pub depth_per_column: f64,
}

impl CaveStats {
    /// Fraction of the rock the caves occupy.
    pub fn void_fraction(&self) -> f64 {
        self.cave_voxels as f64 / self.solid_voxels.max(1) as f64
    }

    /// Share of all cave air sitting in the single largest connected component.
    ///
    /// This is the number that separates tunnels from blobs. One connected cave
    /// system scores near 1.0; N independent spheres score near `1/N`.
    pub fn largest_component_share(&self) -> f64 {
        self.largest_component as f64 / self.cave_voxels.max(1) as f64
    }

    /// Mean voxels per component, another blobbiness indicator: many small
    /// components means stamped objects, few large ones means a connected field.
    pub fn mean_component_size(&self) -> f64 {
        self.cave_voxels as f64 / self.components.max(1) as f64
    }

    /// Surface-to-volume ratio of the cave air, in units of `1/voxel`.
    ///
    /// Reported because it is cheap and it does track *how thick* the cave is:
    /// a one-voxel sheet scores 1.0 (every voxel is surface), a 3-voxel tube
    /// 0.8, and a large solid mass near 0.
    ///
    /// It is deliberately NOT used by [`Self::looks_like_tunnels`]. It was
    /// originally expected to separate tunnels from sheets, and measurement
    /// says the opposite: because it falls as thickness grows, the thinnest
    /// possible sheet scores *higher* than any real tunnel. The assertion that
    /// once encoded that expectation now asserts the true relation. Shape
    /// discrimination is done by `depth_per_column` and `max_column_height`,
    /// which were calibrated against both synthetic shapes.
    pub fn surface_to_volume(&self) -> f64 {
        self.surface_voxels as f64 / self.cave_voxels.max(1) as f64
    }

    /// Whether the caves are shaped like tunnels rather than like a slab.
    ///
    /// Two independent conditions, because either alone can be satisfied by a
    /// degenerate shape:
    ///
    /// * `mean_column_height` small: a tunnel fills only a few voxels of any
    ///   column. A sheet fills its columns top to bottom.
    /// * `max_column_height` small: no single column is filled through a tall
    ///   span, which is the per-node box failure.
    /// * `depth_per_column` above 1: the cave has *thickness* over its
    ///   footprint. A one-voxel sheet is exactly 1 everywhere, and that is the
    ///   only measure that catches it — it is thin in `y` but vast in `xz`, so
    ///   every vertical statistic reads innocent.
    ///
    /// Thresholds are deliberately loose. This is a smoke test for a shape
    /// being catastrophically wrong, and it is calibrated against two synthetic
    /// shapes in the tests rather than against a real cave.
    pub fn looks_like_tunnels(&self) -> bool {
        self.mean_column_height <= COLUMN_HEIGHT_LIMIT
            && self.max_column_height as f64 <= MAX_COLUMN_HEIGHT_LIMIT
            && self.depth_per_column >= MIN_DEPTH_PER_COLUMN
    }

    /// One-line summary for logs and tests.
    pub fn summary(&self) -> String {
        format!(
            "caves: {:.1}% void, {} components, largest {:.1}% of air, trunk {} voxels, {} >=16",
            self.void_fraction() * 100.0,
            self.components,
            self.largest_component_share() * 100.0,
            self.trunk_length,
            self.components_ge_16,
        )
    }
}

/// A cave that fills more than this many voxels of an average column is not a
/// tunnel. MC's noodle caves are 1-5 blocks wide and its branches 2-7, so even
/// generously a tunnel occupies well under ten voxels of any one column.
const COLUMN_HEIGHT_LIMIT: f64 = 10.0;

/// A cave must average at least this many voxels deep across the columns it
/// occupies, or it is a sheet. Exactly 1 is a one-voxel sheet.
const MIN_DEPTH_PER_COLUMN: f64 = 2.0;

/// No single column of a real cave system is filled through more than this.
/// A vertical shaft could exceed it; a network of horizontal tunnels cannot.
const MAX_COLUMN_HEIGHT_LIMIT: f64 = 24.0;

/// Raw voxel grid the statistics are computed over, in world space.
///
/// Kept as a flat `Vec<u8>` of 0/1 rather than a `ChunkGrid` so the flood fill
/// can index it arithmetically without chunk lookups in its inner loop.
struct Mask {
    data: Vec<u8>,
    dims: (usize, usize, usize),
}

impl Mask {
    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        // y varies slowest so a horizontal slice is contiguous.
        z + self.dims.2 * (x + self.dims.0 * y)
    }

    #[inline]
    fn get(&self, x: usize, y: usize, z: usize) -> bool {
        self.data[self.index(x, y, z)] != 0
    }
}

/// Measures the cave shape statistics of a generated world.
///
/// `side` is the band width in chunks on x and z; the band covers the world's
/// full height, so caves at any depth are seen.
pub fn measure(grid: &ChunkGrid, side: i32, mask: CaveMask) -> CaveStats {
    let mask_grid = build_mask(grid, side, mask);
    let (w, h, d) = mask_grid.dims;

    let mut visited = vec![0u8; mask_grid.data.len()];
    let mut components = 0u64;
    let mut components_ge_16 = 0u64;
    let mut largest = 0u64;
    let mut largest_cells: Vec<(usize, usize, usize)> = Vec::new();
    let mut cave_voxels = 0u64;

    let mut queue = VecDeque::new();
    let mut cells = Vec::new();

    for y in 0..h {
        for z in 0..d {
            for x in 0..w {
                if !mask_grid.get(x, y, z) {
                    continue;
                }
                cave_voxels += 1;
                if visited[mask_grid.index(x, y, z)] != 0 {
                    continue;
                }

                components += 1;
                cells.clear();
                queue.clear();
                queue.push_back((x, y, z));
                visited[mask_grid.index(x, y, z)] = 1;

                while let Some((cx, cy, cz)) = queue.pop_front() {
                    cells.push((cx, cy, cz));
                    // 6-connected: two caves touching only at a corner are not
                    // one cave system, and counting them as connected would
                    // overstate connectivity for stamped shapes.
                    for (nx, ny, nz) in neighbours(cx, cy, cz, w, h, d) {
                        if mask_grid.get(nx, ny, nz) && visited[mask_grid.index(nx, ny, nz)] == 0 {
                            visited[mask_grid.index(nx, ny, nz)] = 1;
                            queue.push_back((nx, ny, nz));
                        }
                    }
                }

                if cells.len() as u64 >= 16 {
                    components_ge_16 += 1;
                }
                if cells.len() as u64 > largest {
                    largest = cells.len() as u64;
                    largest_cells.clear();
                    largest_cells.extend_from_slice(&cells);
                }
            }
        }
    }

    // Surface voxels are counted over the whole mask rather than the largest
    // component: a carver that shatters its caves into fragments should not get
    // a better ratio for it.
    let mut surface_voxels = 0u64;
    for y in 0..h {
        for z in 0..d {
            for x in 0..w {
                if !mask_grid.get(x, y, z) {
                    continue;
                }
                // A cave voxel on the grid boundary is treated as exposed,
                // since the measurement box is a crop of a larger world.
                if neighbours(x, y, z, w, h, d).any(|(nx, ny, nz)| !mask_grid.get(nx, ny, nz)) {
                    surface_voxels += 1;
                }
            }
        }
    }

    let (mean_column_height, max_column_height, footprint_columns) =
        column_profile(&largest_cells);
    let depth_per_column = largest as f64 / footprint_columns.max(1) as f64;

    let stats = CaveStats {
        cave_voxels,
        solid_voxels: count_solid(grid, side),
        components,
        largest_component: largest,
        components_ge_16,
        mean_slice_area: mean_slice_area(&largest_cells),
        trunk_length: trunk_length(&largest_cells),
        surface_voxels,
        mean_column_height,
        max_column_height,
        footprint_columns,
        depth_per_column,
    };
    debug_assert!(
        stats.largest_component <= stats.cave_voxels || stats.cave_voxels == 0,
        "largest component cannot exceed the total cave voxels"
    );
    stats
}

fn neighbours(
    x: usize,
    y: usize,
    z: usize,
    w: usize,
    h: usize,
    d: usize,
) -> impl Iterator<Item = (usize, usize, usize)> {
    let mut out = [(0usize, 0usize, 0usize); 6];
    let mut n = 0;
    if x > 0 {
        out[n] = (x - 1, y, z);
        n += 1;
    }
    if x + 1 < w {
        out[n] = (x + 1, y, z);
        n += 1;
    }
    if y > 0 {
        out[n] = (x, y - 1, z);
        n += 1;
    }
    if y + 1 < h {
        out[n] = (x, y + 1, z);
        n += 1;
    }
    if z > 0 {
        out[n] = (x, y, z - 1);
        n += 1;
    }
    if z + 1 < d {
        out[n] = (x, y, z + 1);
        n += 1;
    }
    out.into_iter().take(n)
}

/// Copies the requested voxels out of the chunk grid into one flat mask.
fn build_mask(grid: &ChunkGrid, side: i32, mask: CaveMask) -> Mask {
    let w = (side * 16) as usize;
    let h = (grid.size().1 * 16) as usize;
    let d = (side * 16) as usize;
    let mut data = vec![0u8; w * h * d];

    // Laid out as z + d * (x + w * y): must match `Mask::index` exactly, or the
    // two disagree and the flood fill reads a scrambled grid. `mask_layout_matches_mask_index`
    // in the tests below pins that down.
    let at = |x: usize, y: usize, z: usize| z + d * (x + w * y);

    let (_, sy, _) = grid.size();
    for cy in 0..sy as i32 {
        for cz in 0..side {
            for cx in 0..side {
                let pos = ChunkPos::new(cx, cy, cz);
                let chunk = grid.get(pos);
                let (ox, oy, oz) = pos.origin();
                for i in 0..CHUNK_VOLUME as u16 {
                    let idx = unsafe { VoxelIndex::from_raw_unchecked(i) };
                    if !chunk.block(idx).is_air() {
                        continue;
                    }
                    let (mx, my, mz) = (
                        (ox + idx.x() as i32) as usize,
                        (oy + idx.y() as i32) as usize,
                        (oz + idx.z() as i32) as usize,
                    );
                    if mx >= w || my >= h || mz >= d {
                        continue;
                    }
                    let is_cave = match mask {
                        CaveMask::AllAir => true,
                        CaveMask::Underground => !open_to_sky(grid, pos, idx),
                    };
                    if is_cave {
                        data[at(mx, my, mz)] = 1;
                    }
                }
            }
        }
    }

    Mask {
        data,
        dims: (w, h, d),
    }
}

fn count_solid(grid: &ChunkGrid, side: i32) -> u64 {
    let mut solid = 0u64;
    let (_, sy, _) = grid.size();
    for cy in 0..sy as i32 {
        for cz in 0..side {
            for cx in 0..side {
                let chunk = grid.get(ChunkPos::new(cx, cy, cz));
                for i in 0..CHUNK_VOLUME as u16 {
                    let idx = unsafe { VoxelIndex::from_raw_unchecked(i) };
                    if !chunk.block(idx).is_air() {
                        solid += 1;
                    }
                }
            }
        }
    }
    solid
}

/// True if the voxel has a clear column straight up to the world ceiling.
///
/// A column test rather than a flood fill, matching `report::carved_voxel_count`
/// so the two agree on what "a cave" means.
fn open_to_sky(grid: &ChunkGrid, pos: ChunkPos, index: VoxelIndex) -> bool {
    let (ox, oy, oz) = pos.origin();
    let (wx, mut wy, wz) = (
        ox + index.x() as i32,
        oy + index.y() as i32 + 1,
        oz + index.z() as i32,
    );
    let (sx, sy, sz) = grid.size();
    let top = grid.size().1 as i32 * 16;
    while wy < top {
        let cpos = ChunkPos::from_world(wx, wy, wz);
        if cpos.x < 0
            || cpos.y < 0
            || cpos.z < 0
            || cpos.x >= sx as i32
            || cpos.y >= sy as i32
            || cpos.z >= sz as i32
        {
            return true;
        }
        let chunk = grid.get(cpos);
        let local = VoxelIndex::from_parts(
            (wx - cpos.x * 16) as u8,
            (wy - cpos.y * 16) as u8,
            (wz - cpos.z * 16) as u8,
        );
        if !chunk.block(local).is_air() {
            return false;
        }
        wy += 1;
    }
    true
}

/// Mean filled height per occupied `(x, z)` column, and the tallest such
/// column.
///
/// The pair is what catches a flat-ceiling slab. A tunnel occupies a few voxels
/// of each column it passes through and its ceiling follows the tunnel, so both
/// numbers stay small. A per-node carver that gives every column of a footprint
/// the same `y` interval fills each column to the same height, so the mean
/// climbs and the fully-filled share jumps.
fn column_profile(cells: &[(usize, usize, usize)]) -> (f64, u64, u64) {
    if cells.is_empty() {
        return (0.0, 0, 0);
    }
    let mut per_column: std::collections::BTreeMap<(usize, usize), (u64, usize, usize)> =
        std::collections::BTreeMap::new();
    for &(x, y, z) in cells {
        let entry = per_column.entry((x, z)).or_insert((0, usize::MAX, 0));
        entry.0 += 1;
        entry.1 = entry.1.min(y);
        entry.2 = entry.2.max(y);
    }
    let total = cells.len() as f64;
    let columns = per_column.len() as f64;
    // A column is "full" when it is filled from floor to ceiling of the cave's
    // own vertical extent, which is the flat-ceiling signature.
    let tallest = per_column
        .values()
        .map(|&(count, _, _)| count)
        .max()
        .unwrap_or(0);
    (total / columns.max(1.0), tallest, columns as u64)
}

/// Mean number of largest-component voxels per occupied y slice.
///
/// A tunnel contributes a small constant per slice; a room contributes an area
/// that grows with its width, so this separates the two.
fn mean_slice_area(cells: &[(usize, usize, usize)]) -> f64 {
    if cells.is_empty() {
        return 0.0;
    }
    let mut per_slice = std::collections::BTreeMap::new();
    for &(_, y, _) in cells {
        *per_slice.entry(y).or_insert(0u64) += 1;
    }
    cells.len() as f64 / per_slice.len().max(1) as f64
}

/// Longest axis-aligned extent of the largest component.
///
/// Minecraft's trunks run 85-112 blocks, so this is the statistic that catches
/// a carver whose "caves" are all small.
fn trunk_length(cells: &[(usize, usize, usize)]) -> u64 {
    if cells.is_empty() {
        return 0;
    }
    let mut min = [usize::MAX; 3];
    let mut max = [0usize; 3];
    for &(x, y, z) in cells {
        for (axis, v) in [x, y, z].into_iter().enumerate() {
            min[axis] = min[axis].min(v);
            max[axis] = max[axis].max(v);
        }
    }
    (0..3).map(|a| (max[a] - min[a] + 1) as u64).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::BlockId;

    /// Builds a small grid where the caller paints air voxels directly.
    fn grid_with(size: u32, paint: impl Fn(i32, i32, i32) -> bool) -> ChunkGrid {
        let mut grid = ChunkGrid::new(size, size, size);
        for pos in grid.positions().collect::<Vec<_>>() {
            let chunk = grid.get_mut(pos);
            chunk.fill(BlockId::STONE);
            let (ox, oy, oz) = pos.origin();
            for i in 0..CHUNK_VOLUME as u16 {
                let idx = unsafe { VoxelIndex::from_raw_unchecked(i) };
                let (wx, wy, wz) = (
                    ox + idx.x() as i32,
                    oy + idx.y() as i32,
                    oz + idx.z() as i32,
                );
                if paint(wx, wy, wz) {
                    chunk.set_block(idx, BlockId::AIR);
                }
            }
        }
        grid
    }

    /// A straight tunnel along x, `thickness` voxels wide, buried under stone.
    fn tunnel_grid(size: u32, thickness: i32) -> ChunkGrid {
        grid_with(size, |x, y, z| {
            let (cy, cz) = (y - 16, z - 16);
            (16..32).contains(&y)
                && cy.abs() < thickness
                && cz.abs() < thickness
                && (0..size as i32 * 16).contains(&x)
        })
    }

    /// One isolated cube per chunk: the "stamped blobs" failure mode.
    fn blobs_grid(size: u32) -> ChunkGrid {
        grid_with(size, |x, y, z| {
            let (lx, ly, lz) = (x.rem_euclid(16), y.rem_euclid(16), z.rem_euclid(16));
            (8..12).contains(&lx) && (8..12).contains(&ly) && (8..12).contains(&lz)
        })
    }

    /// The one structural invariant the whole module rests on: the closure that
    /// writes the mask and `Mask::index` that reads it must agree. They were
    /// written with different layouts once, and a mismatch scrambles the grid
    /// silently rather than failing.
    #[test]
    fn mask_layout_matches_mask_index() {
        // Both voxels must lie inside a 2x2x2 chunk world (32 voxels per axis)
        // or `build_mask` drops them on its bounds check and the test asserts
        // nothing. A first version of this test painted y=40 and silently
        // tested only one voxel.
        let (a, b) = ((5, 3, 7), (30, 20, 25));
        let grid = grid_with(2, |x, y, z| (x, y, z) == a || (x, y, z) == b);
        let mask = build_mask(&grid, 2, CaveMask::AllAir);
        let (w, h, d) = mask.dims;
        for (x, y, z) in [a, b] {
            let (x, y, z) = (x as usize, y as usize, z as usize);
            assert!(
                x < w && y < h && z < d,
                "test premise: {x},{y},{z} must be inside {w}x{h}x{d}"
            );
            assert!(
                mask.get(x, y, z),
                "voxel {x},{y},{z} not where the mask reader looks for it"
            );
        }
    }

    /// A single long tunnel must read as one component covering all its air.
    #[test]
    fn a_connected_tunnel_is_one_component() {
        let grid = tunnel_grid(2, 2);
        let stats = measure(&grid, 2, CaveMask::Underground);
        assert!(stats.cave_voxels > 0, "tunnel grid produced no cave air");
        assert_eq!(
            stats.components, 1,
            "a straight tunnel must be a single component: {}",
            stats.summary()
        );
        assert_eq!(
            stats.largest_component_share(),
            1.0,
            "all of a connected tunnel's air is in its one component"
        );
    }

    /// The statistic this module exists for: blobs must score far lower on
    /// connectivity than a tunnel does, at similar void fraction. If this ever
    /// passes for blobs, the yardstick cannot tell MC caves from spheres and
    /// the whole comparison is worthless.
    #[test]
    fn blobs_are_less_connected_than_a_tunnel() {
        let tunnel = measure(&tunnel_grid(4, 2), 4, CaveMask::Underground);
        let blobs = measure(&blobs_grid(4), 4, CaveMask::Underground);

        assert!(blobs.cave_voxels > 0, "blob grid produced no cave air");
        assert!(
            blobs.components > tunnel.components,
            "blobs ({}) should be more fragmented than a tunnel ({})",
            blobs.components,
            tunnel.components
        );
        assert!(
            blobs.largest_component_share() < tunnel.largest_component_share(),
            "blobs score {:.3} connectivity, tunnel {:.3}; the metric must \
             separate them or it cannot judge MC-likeness",
            blobs.largest_component_share(),
            tunnel.largest_component_share()
        );
    }

    /// A tunnel spanning the whole band must report a long trunk; a single
    /// small blob must not.
    #[test]
    fn trunk_length_separates_long_from_small() {
        let tunnel = measure(&tunnel_grid(4, 2), 4, CaveMask::Underground);
        let blobs = measure(&blobs_grid(4), 4, CaveMask::Underground);
        assert!(
            tunnel.trunk_length > 16,
            "a tunnel across 4 chunks should have a long extent, got {}",
            tunnel.trunk_length
        );
        assert!(
            tunnel.trunk_length > blobs.trunk_length,
            "tunnel extent {} should exceed blob extent {}",
            tunnel.trunk_length,
            blobs.trunk_length
        );
    }

    /// Sky is not a cave. Counting it would let any carver look connected by
    /// being adjacent to the surface.
    #[test]
    fn open_to_sky_air_is_not_counted_as_a_cave() {
        // A full column of air down the middle of the world, reaching the top.
        let grid = grid_with(2, |x, y, z| x.rem_euclid(16) == 8 && z.rem_euclid(16) == 8 && y > 0);
        let underground = measure(&grid, 2, CaveMask::Underground);
        let all_air = measure(&grid, 2, CaveMask::AllAir);
        assert!(
            all_air.cave_voxels > underground.cave_voxels,
            "the sky-reaching column must be excluded from the underground mask"
        );
    }

    /// An empty world must not panic or divide by zero anywhere.
    #[test]
    fn a_world_with_no_caves_reports_zeroes() {
        let grid = grid_with(2, |_, _, _| false);
        let stats = measure(&grid, 2, CaveMask::Underground);
        assert_eq!(stats.cave_voxels, 0);
        assert_eq!(stats.components, 0);
        assert_eq!(stats.largest_component_share(), 0.0);
        assert_eq!(stats.void_fraction(), 0.0);
    }

    /// The measurement must be deterministic, or it cannot be asserted in a
    /// test and cannot be compared across carvers.
    #[test]
    fn measurement_is_deterministic() {
        let grid = tunnel_grid(2, 2);
        let a = measure(&grid, 2, CaveMask::Underground);
        let b = measure(&grid, 2, CaveMask::Underground);
        assert_eq!(a, b, "two runs over the same grid disagreed");
    }

    /// 6-connectivity, not 26: two cubes touching only at a corner are two
    /// caves, and counting them as one would overstate stamped carvers.
    #[test]
    fn corner_touching_cubes_are_separate_components() {
        let grid = grid_with(1, |x, y, z| {
            (x, y, z) == (5, 5, 5) || (x, y, z) == (6, 6, 6)
        });
        let stats = measure(&grid, 1, CaveMask::Underground);
        assert_eq!(
            stats.components, 2,
            "corner-adjacent cubes must not count as one component"
        );
    }
}

#[cfg(test)]
mod shape_tests {
    use super::*;
    use crate::domain::BlockId;

    fn grid_with(size: u32, paint: impl Fn(i32, i32, i32) -> bool) -> ChunkGrid {
        let mut grid = ChunkGrid::new(size, size, size);
        for pos in grid.positions().collect::<Vec<_>>() {
            let chunk = grid.get_mut(pos);
            chunk.fill(BlockId::STONE);
            let (ox, oy, oz) = pos.origin();
            for i in 0..CHUNK_VOLUME as u16 {
                let idx = unsafe { VoxelIndex::from_raw_unchecked(i) };
                let (wx, wy, wz) = (
                    ox + idx.x() as i32,
                    oy + idx.y() as i32,
                    oz + idx.z() as i32,
                );
                if paint(wx, wy, wz) {
                    chunk.set_block(idx, BlockId::AIR);
                }
            }
        }
        grid
    }

    /// The failure this metric exists for: a carver that fills every column of
    /// a region to the same height makes a slab, and a slab scored 99.9%
    /// connectivity while being plainly not a cave.
    #[test]
    fn a_flat_slab_is_not_tunnels_and_a_tube_is() {
        // A slab: one thin horizontal sheet across the whole world.
        let slab = grid_with(4, |_, y, _| y == 30);
        let slab_stats = measure(&slab, 4, CaveMask::Underground);
        println!(
            "slab: mean_col {:.1} max_col {} sv {:.3} tunnels={}",
            slab_stats.mean_column_height,
            slab_stats.max_column_height,
            slab_stats.surface_to_volume(),
            slab_stats.looks_like_tunnels()
        );
        assert!(
            !slab_stats.looks_like_tunnels(),
            "a flat slab must not pass the tunnel shape test: {}",
            slab_stats.summary()
        );

        // A tube: a thin winding line, 3 voxels tall.
        let tube = grid_with(4, |x, y, z| {
            (16..19).contains(&y) && (z - (x / 2 + 16)).abs() < 3
        });
        let tube_stats = measure(&tube, 4, CaveMask::Underground);
        println!(
            "tube: mean_col {:.1} max_col {} sv {:.3} tunnels={}",
            tube_stats.mean_column_height,
            tube_stats.max_column_height,
            tube_stats.surface_to_volume(),
            tube_stats.looks_like_tunnels()
        );
        assert!(
            tube_stats.looks_like_tunnels(),
            "a 3-voxel-tall tube must pass the tunnel shape test: {}",
            tube_stats.summary()
        );
    }

    /// The relation surface-to-volume actually has: it falls as the cave gets
    /// thicker, so a one-voxel sheet scores 1.0 and a mass scores near 0.
    ///
    /// This test was written the other way round, asserting that a tube exposes
    /// *more* surface than a slab. Measurement contradicted it, and rather than
    /// deleting the check the claim was corrected, because the wrong version
    /// would have justified using the metric to judge shape.
    #[test]
    fn surface_to_volume_falls_as_the_cave_gets_thicker() {
        let slab = measure(&grid_with(4, |_, y, _| y == 30), 4, CaveMask::Underground);
        let tube = graph_tube(4);
        println!(
            "slab sv {:.3} vs tube sv {:.3}",
            slab.surface_to_volume(),
            tube.surface_to_volume()
        );
        assert!(
            tube.surface_to_volume() < slab.surface_to_volume(),
            "a 3-voxel tube (sv {:.3}) should expose LESS surface per voxel than a \
             one-voxel sheet (sv {:.3}), because thinner means more surface",
            tube.surface_to_volume(),
            slab.surface_to_volume()
        );
    }

    fn graph_tube(size: u32) -> CaveStats {
        measure(
            &grid_with(size, |x, y, z| {
                (16..19).contains(&y) && (z - (x / 2 + 16)).abs() < 3
            }),
            size as i32,
            CaveMask::Underground,
        )
    }
}
