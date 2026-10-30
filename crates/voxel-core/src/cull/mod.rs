//! Culling (`PROJECT.md` §4.3).
//!
//! Three independent rejections, cheapest first, so the expensive test runs on
//! as few chunks as possible:
//!
//! 1. **Empty and enclosed skip** — one bool read. A chunk that is all air, or
//!    all opaque solid with no exposed face, cannot contribute geometry.
//! 2. **Frustum test** — a chunk whose bounding box is outside the view volume.
//! 3. **Distance test** — beyond the configured render radius.
//!
//! Deliberately no occlusion queries or hardware queries: this POC has no
//! render graph to hang them on, and the flags in step 1 already remove the
//! subterranean bulk that matters.

mod frustum;

pub use frustum::{Frustum, Plane};

use crate::domain::{CHUNK_SIZE, ChunkPos};
use crate::world::ChunkGrid;

/// Why a chunk was skipped, so a caller can report the breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CullStats {
    pub total: u64,
    pub empty: u64,
    pub enclosed: u64,
    pub outside_frustum: u64,
    pub too_far: u64,
    pub drawn: u64,
}

impl CullStats {
    pub fn skipped(&self) -> u64 {
        self.empty + self.enclosed + self.outside_frustum + self.too_far
    }

    /// Human-readable one-liner for the debug overlay.
    pub fn summary(&self) -> String {
        format!(
            "{} chunks: {} drawn, {} skipped ({} empty, {} enclosed, {} offscreen, {} far)",
            self.total,
            self.drawn,
            self.skipped(),
            self.empty,
            self.enclosed,
            self.outside_frustum,
            self.too_far
        )
    }
}

/// A chunk the renderer should draw, with the world-space centre for sorting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisibleChunk {
    pub pos: ChunkPos,
    pub centre: [f32; 3],
    /// Squared distance to the camera, for front-to-back ordering.
    pub distance_sq: f32,
}

/// Selects the chunks to draw.
///
/// `radius` is in chunks; `0` means no distance limit.
pub fn select_visible(
    grid: &ChunkGrid,
    frustum: &Frustum,
    camera: [f32; 3],
    radius: i32,
) -> (Vec<VisibleChunk>, CullStats) {
    let mut out = Vec::new();
    let mut stats = CullStats::default();
    let radius_sq = (radius as f32 * CHUNK_SIZE as f32).powi(2);

    for (pos, chunk) in grid.iter() {
        stats.total += 1;

        // Cheapest rejections first: the cached flags need no arithmetic and
        // remove the whole subterranean and sky bulk of a typical world.
        if chunk.empty {
            stats.empty += 1;
            continue;
        }
        if chunk.enclosed {
            stats.enclosed += 1;
            continue;
        }

        let (ox, oy, oz) = pos.origin();
        let half = CHUNK_SIZE as f32 * 0.5;
        let centre = [ox as f32 + half, oy as f32 + half, oz as f32 + half];

        let distance_sq = {
            let d = [
                centre[0] - camera[0],
                centre[1] - camera[1],
                centre[2] - camera[2],
            ];
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        };

        if radius > 0 && distance_sq > radius_sq {
            stats.too_far += 1;
            continue;
        }

        if !frustum.intersects_chunk(pos) {
            stats.outside_frustum += 1;
            continue;
        }

        stats.drawn += 1;
        out.push(VisibleChunk {
            pos,
            centre,
            distance_sq,
        });
    }

    // Front to back, so opaque geometry fills depth first and any future
    // transparency pass can early-out on depth.
    out.sort_by(|a, b| {
        a.distance_sq
            .partial_cmp(&b.distance_sq)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pos.cmp(&b.pos))
    });

    (out, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::BlockId;
    use crate::domain::VoxelIndex;

    /// A grid where every chunk is a normal surface chunk, so culling is
    /// exercised on the frustum and distance paths rather than the flag paths.
    fn open_grid(size: (u32, u32, u32)) -> ChunkGrid {
        let mut grid = ChunkGrid::new(size.0, size.1, size.2);
        let positions: Vec<_> = grid.positions().collect();
        for pos in positions {
            let c = grid.get_mut(pos);
            // One voxel, so the chunk is neither empty nor enclosed.
            c.set_block(VoxelIndex::from_parts(0, 0, 0), BlockId::STONE);
            c.refresh_flags();
        }
        grid
    }

    #[test]
    fn empty_and_enclosed_chunks_are_skipped_without_geometry_work() {
        let mut grid = ChunkGrid::new(2, 1, 2);
        let positions: Vec<_> = grid.positions().collect();

        // One all-air, one all-solid, one surface, one single voxel.
        let mut iter = positions.clone().into_iter();
        let air = iter.next().unwrap();
        grid.get_mut(air).refresh_flags();

        let solid = iter.next().unwrap();
        grid.get_mut(solid).fill(BlockId::STONE);
        grid.get_mut(solid).refresh_flags();

        let surface = iter.next().unwrap();
        grid.get_mut(surface)
            .set_block(VoxelIndex::from_parts(0, 0, 0), BlockId::STONE);
        grid.get_mut(surface).refresh_flags();

        let single = iter.next().unwrap();
        grid.get_mut(single)
            .set_block(VoxelIndex::from_parts(8, 8, 8), BlockId::GRASS);
        grid.get_mut(single).refresh_flags();

        let frustum = Frustum::everything();
        let (visible, stats) = select_visible(&grid, &frustum, [0.0; 3], 0);

        assert_eq!(stats.total, 4);
        assert_eq!(stats.empty, 1, "all-air chunk must be skipped");
        assert_eq!(stats.enclosed, 1, "all-solid chunk must be skipped");
        assert_eq!(stats.drawn, 2, "the two mixed chunks must be drawn");
        assert_eq!(visible.len(), 2);
    }

    #[test]
    fn everything_frustum_draws_all_non_skippable_chunks() {
        let grid = open_grid((3, 1, 3));
        let (visible, stats) = select_visible(&grid, &Frustum::everything(), [0.0; 3], 0);
        assert_eq!(stats.drawn, 9);
        assert_eq!(visible.len(), 9);
        assert_eq!(stats.skipped(), 0);
    }

    #[test]
    fn distance_limit_rejects_far_chunks() {
        let grid = open_grid((5, 1, 5));
        // Camera at the origin chunk's low corner, radius 2 chunks = 32 voxels.
        let (visible, stats) = select_visible(&grid, &Frustum::everything(), [8.0; 3], 2);
        assert!(stats.too_far > 0, "some chunks must be beyond the radius");
        assert!(visible.len() < 25, "the radius must reject something");
        for v in &visible {
            assert!(
                v.distance_sq <= (2.0 * CHUNK_SIZE as f32).powi(2),
                "kept a chunk beyond the radius: {v:?}"
            );
        }
    }

    #[test]
    fn zero_radius_means_no_distance_limit() {
        let grid = open_grid((4, 1, 4));
        let (visible, stats) = select_visible(&grid, &Frustum::everything(), [0.0; 3], 0);
        assert_eq!(stats.too_far, 0);
        assert_eq!(visible.len(), 16);
    }

    #[test]
    fn ordering_is_front_to_back() {
        let grid = open_grid((3, 1, 3));
        let camera = [0.0, 0.0, 0.0];
        let (visible, _) = select_visible(&grid, &Frustum::everything(), camera, 0);
        for pair in visible.windows(2) {
            assert!(
                pair[0].distance_sq <= pair[1].distance_sq,
                "visible list is not sorted by distance"
            );
        }
    }

    #[test]
    fn a_narrow_frustum_rejects_off_screen_chunks() {
        let grid = open_grid((5, 1, 5));
        // Looking straight down +X from inside the world: chunks behind and to
        // the sides must be rejected.
        let camera = [8.0, 8.0, 8.0];
        let frustum = Frustum::perspective(
            camera,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            std::f32::consts::FRAC_PI_2,
            1.0,
            0.1,
            1000.0,
        );
        let (visible, stats) = select_visible(&grid, &frustum, camera, 0);
        assert!(
            stats.outside_frustum > 0,
            "a 90-degree frustum must reject something in a 5x1x5 grid"
        );
        assert!(visible.len() < 25);

        // Everything kept must be in front of the camera along +X.
        for v in &visible {
            assert!(
                v.centre[0] >= camera[0] - CHUNK_SIZE as f32,
                "kept a chunk behind the camera: {v:?}"
            );
        }
    }

    #[test]
    fn stats_account_for_every_chunk_exactly_once() {
        let grid = open_grid((4, 1, 4));
        let frustum = Frustum::everything();
        let (_, stats) = select_visible(&grid, &frustum, [0.0; 3], 1);
        assert_eq!(
            stats.drawn + stats.skipped(),
            stats.total,
            "every chunk must be counted once"
        );
    }

    #[test]
    fn summary_mentions_every_category() {
        let s = CullStats {
            total: 10,
            empty: 1,
            enclosed: 2,
            outside_frustum: 3,
            too_far: 4,
            drawn: 0,
        };
        let text = s.summary();
        for needle in ["10 chunks", "empty", "enclosed", "offscreen", "far"] {
            assert!(text.contains(needle), "summary missing {needle:?}: {text}");
        }
        assert_eq!(s.skipped(), 10);
    }
}
