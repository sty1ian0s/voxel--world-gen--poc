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

    let stats = CaveStats {
        cave_voxels,
        solid_voxels: count_solid(grid, side),
        components,
        largest_component: largest,
        components_ge_16,
        mean_slice_area: mean_slice_area(&largest_cells),
        trunk_length: trunk_length(&largest_cells),
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
