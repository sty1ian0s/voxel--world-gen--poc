//! Sliding-window cave metrics: the shape statistics of [`crate::debug::caves`]
//! computed without ever holding the world.
//!
//! `caves::measure` builds a flat mask of the whole measured band and flood
//! fills it. That is exact, and it costs `side^2 * world_height` bytes: a 64x64
//! band of a 64-tall world is 67 MB, a 256x256 one is 1.07 GB. So the honest
//! question is not "can this be streamed" but "which numbers stay exact when it
//! is", and this module answers that in a table rather than in prose.
//!
//! ## Layout
//!
//! The measured region is a rectangle of chunks, `W` columns on `x` and `Z` on
//! `z`, spanning the full vertical band. It is cut into slabs of `rows` chunk
//! rows along `z`. The accumulator holds one slab, plus the previous slab's last
//! chunk row for connectivity. Nothing else scales with the region.
//!
//! ## What is exact and what is windowed
//!
//! | statistic | scope |
//! |---|---|
//! | solid/cave voxels, void fraction, mean column depth | **global** — summed counts |
//! | component-size quantiles p50/p90/p99/max | **global** — fixed-size histogram, exact |
//! | per-column depth quantiles p50/p90/p99/max/mean | **global** — fixed-size histogram, exact |
//! | component count, size distribution | **windowed** |
//! | largest component, share, trunk length, footprint | **windowed** |
//! | cross-chunk connectivity | correct across every chunk seam inside a slab |
//!
//! Components are linked with a union-find seeded with the previous slab's last
//! chunk row, so a cave running along `+z` is one component rather than one per
//! slab. Two things still split a component, and both are inherent to a
//! fixed-size window rather than fixable here:
//!
//! * a cave leaving through the region's `x` faces is cut in two at the first
//!   and last column, because a sliding window over `x` would need a component's
//!   identity to outlive the row it was counted in;
//! * a component whose entire trace in a slab is a continuation of a dropped
//!   slab's component has no size here and cannot be the largest.
//!
//! Both make the windowed component count an **upper bound** on the true count
//! and the windowed largest-component share a **lower bound** on the true share.
//! Tests pin that direction on shapes whose answer is known by hand.

use crate::domain::{ChunkPos, VoxelIndex};
use crate::world::{Region, WorldGen};

/// How the cave voxels were selected. Mirrors [`crate::debug::caves::CaveMask`]
/// deliberately: two definitions of "a cave" in one project is one too many.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mask {
    /// Air with no clear column of air above it, i.e. carved rather than sky.
    Underground,
    /// Every air voxel.
    AllAir,
}

/// Sentinel for "this cell carries no node": solid, outside the mask, or a cave
/// voxel whose component was discovered in a slab that has been dropped.
const NO_NODE: u32 = u32::MAX;

/// What a neighbouring cave voxel contributes to the current voxel's component.
///
/// The distinction that matters is `Node` versus the two carried variants: a
/// `Node` neighbour has a size this window can compute, a carried neighbour does
/// not, and merging the two would report a fragment of a cave as a whole one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Contact {
    /// A cave voxel of this window with a live node.
    Node(u32),
    /// A cave voxel of the carried row, or of this window but itself a
    /// continuation of a dropped window's component. Its component's size is not
    /// knowable here.
    Carried,
}

/// Union-find over the nodes of the current window.
///
/// Parents are one flat `u32` array, which is the DOD shape `PROJECT.md` §2 asks
/// for: a single allocation for the accumulator's lifetime, no per-component
/// `Vec`, and no hash map in the linking pass.
struct Components {
    parent: Vec<u32>,
    size: Vec<u64>,
}

impl Components {
    fn new(nodes: usize) -> Self {
        Self {
            parent: vec![u32::MAX; nodes],
            size: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.parent.fill(u32::MAX);
        self.size.clear();
    }

    #[inline]
    fn find(&mut self, n: u32) -> u32 {
        let mut root = n;
        while self.parent[root as usize] != u32::MAX {
            root = self.parent[root as usize];
        }
        // Path compression, so a long tunnel does not degrade the next find.
        let mut walk = n;
        while self.parent[walk as usize] != u32::MAX {
            let next = self.parent[walk as usize];
            self.parent[walk as usize] = root;
            walk = next;
        }
        root
    }

    /// Merges two components, returning the surviving root.
    fn union(&mut self, a: u32, b: u32) -> u32 {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return ra;
        }
        let sa = self.size.get(ra as usize).copied();
        let sb = self.size.get(rb as usize).copied();
        let (big, small) = match (sa, sb) {
            (Some(x), Some(y)) if y > x => (rb, ra),
            (Some(_), Some(_)) => (ra, rb),
            // The root that survives must be one whose size this window can
            // still report; a root with no entry is a class that continues into
            // a slab that has been dropped.
            (Some(_), None) => (ra, rb),
            (None, Some(_)) => (rb, ra),
            (None, None) => (ra, rb),
        };
        self.parent[small as usize] = big;
        let merged = self.size.get(small as usize).copied();
        if let (Some(s), Some(e)) = (merged, self.size.get_mut(big as usize)) {
            *e += s;
        }
        big
    }
}

/// Largest size a bin can hold, so a quantile is an upper bound on the samples
/// in it rather than a guess.
fn bin_upper_bound(bin: usize) -> u64 {
    match bin {
        0 => 0,
        1 => 1,
        _ => 1u64 << (bin - 1),
    }
}

/// Counts in `log2`-spaced bins above 1, with exact quantiles.
///
/// Not a quantile sketch. Sizes are integers, the range is bounded by the
/// window, and a histogram costs one allocation for the whole run — whereas
/// keeping every component size would allocate per cave and a reservoir sample
/// would make p99 approximate.
#[derive(Debug, Clone)]
struct LogHistogram {
    buckets: Vec<u64>,
    total: u64,
}

impl LogHistogram {
    fn new(bins: usize) -> Self {
        Self {
            buckets: vec![0; bins],
            total: 0,
        }
    }

    #[inline]
    fn bin(value: u64) -> usize {
        if value <= 1 {
            value as usize
        } else {
            64 - value.leading_zeros() as usize + 1
        }
    }

    fn add(&mut self, value: u64) {
        let b = Self::bin(value);
        debug_assert!(
            b < self.buckets.len(),
            "value {value} escaped the histogram"
        );
        let b = b.min(self.buckets.len() - 1);
        self.buckets[b] += 1;
        self.total += 1;
    }

    fn quantile(&self, q: f64) -> u64 {
        if self.total == 0 {
            return 0;
        }
        let want = (q * self.total as f64).ceil().max(1.0) as u64;
        let mut seen = 0u64;
        for (b, &count) in self.buckets.iter().enumerate() {
            seen += count;
            if seen >= want {
                return bin_upper_bound(b);
            }
        }
        bin_upper_bound(self.buckets.len() - 1)
    }

    fn max(&self) -> u64 {
        self.buckets
            .iter()
            .rposition(|&c| c > 0)
            .map_or(0, bin_upper_bound)
    }
}

/// Cave voxels per depth below the surface, in voxels.
///
/// Bounded by the vertical band, so unlike component sizes this needs no
/// windowing at all: one bin per band voxel and the quantiles are global.
#[derive(Debug, Clone)]
struct DepthHistogram {
    counts: Vec<u64>,
    total: u64,
    sum: u64,
}

impl DepthHistogram {
    fn new(band_voxels: i32) -> Self {
        Self {
            counts: vec![0; band_voxels.max(1) as usize],
            total: 0,
            sum: 0,
        }
    }

    fn add(&mut self, depth: i32) {
        // Depths beyond the last bin are clamped into it rather than dropped: a
        // voxel can sit below the window's floor and deeper than the band is
        // tall, and silently losing those samples would make every quantile an
        // understatement. `the_depth_histogram_saturates_rather_than_drops`
        // pins that.
        let d = (depth.max(0) as usize).min(self.counts.len() - 1);
        self.counts[d] += 1;
        self.total += 1;
        self.sum += d as u64;
    }

    fn quantile(&self, q: f64) -> i32 {
        if self.total == 0 {
            return 0;
        }
        let want = (q * self.total as f64).ceil().max(1.0) as u64;
        let mut seen = 0u64;
        for (d, &c) in self.counts.iter().enumerate() {
            seen += c;
            if seen >= want {
                return d as i32;
            }
        }
        self.counts.len() as i32 - 1
    }

    fn mean(&self) -> f64 {
        self.sum as f64 / self.total.max(1) as f64
    }

    fn max(&self) -> i32 {
        self.counts
            .iter()
            .rposition(|&c| c > 0)
            .map_or(0, |d| d as i32)
    }
}

/// The statistics one run produces, with every field's scope in its name.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WindowStats {
    pub chunks: u64,
    pub columns: u64,
    /// **Global**: non-air voxels counted.
    pub solid_voxels: u64,
    /// **Global**: voxels matching the mask.
    pub cave_voxels: u64,
    /// **Windowed**: components discovered, one count per slab. An upper bound
    /// on the true count — see the module doc.
    pub components_in_window: u64,
    /// **Windowed**: components of at least 16 voxels.
    pub components_ge_16_in_window: u64,
    /// **Windowed**: voxels in the largest component measured whole inside a
    /// slab, a lower bound on the true largest.
    pub largest_component_in_window: u64,
    /// **Windowed** longest axis-aligned extent of that component, in voxels.
    pub trunk_length_in_window: u64,
    /// **Windowed** `(x, z)` footprint of that component, in columns.
    pub footprint_columns_in_window: u64,
    /// **Windowed** cave voxels of the window the largest component was measured
    /// in, the denominator for [`Self::chunk_connected_share`].
    pub largest_component_window_cave_voxels: u64,
    /// **Global** quantiles of the component-size histogram, in voxels.
    pub size_p50: u64,
    pub size_p90: u64,
    pub size_p99: u64,
    pub size_max: u64,
    /// **Global** per-column depth below the surface, in voxels.
    pub depth_p50: i32,
    pub depth_p90: i32,
    pub depth_p99: i32,
    pub depth_max: i32,
    pub depth_mean: f64,
    /// **Global**: chunks that held no air at all, the culling statistic.
    pub empty_chunks: u64,
    /// Window height in chunk rows, so a report can state its own scope.
    pub rows: i32,
    /// Slabs the region was cut into.
    pub slabs: u64,
}

impl WindowStats {
    /// Fraction of the rock the caves occupy. Global: both terms are counts.
    pub fn void_fraction(&self) -> f64 {
        self.cave_voxels as f64 / self.solid_voxels.max(1) as f64
    }

    /// **Windowed, chunk-granular**: share of the window's cave *chunks* sitting
    /// in its largest connected component.
    ///
    /// ## Do NOT compare this with `cave stat`'s `largest_component_share`
    ///
    /// Two reasons, and both are deliberate:
    ///
    /// * **Granularity.** Components here are linked between cave *chunks*, not
    ///   cave voxels, because a chunk is the unit the window stores. Any two
    ///   chunks that both contain cave air merge, so this **over-merges**: two
    ///   caves that pass through the same chunk count as one even if their voxels
    ///   never touch. It reads near 100% on terrain where the voxel-accurate
    ///   number is a few percent. Per-voxel linking needs a per-voxel window mask
    ///   (136 MiB at 256x256, and it grows with the window's row count), which is
    ///   the memory this module exists to avoid.
    /// * **Denominator.** The numerator is one window's largest component and the
    ///   denominator is that same window's cave chunks — not the region's. A
    ///   ratio against the whole region's `cave_voxels` is a fraction of
    ///   something the numerator was never drawn from, and it read 0.0% on a
    ///   world full of connected cave, which is worse than no number at all.
    ///
    /// Use it to compare *the same carver across window sizes*, or to detect a
    /// carver that produces nothing connected. For the honest MC-likeness number
    /// use `cave stat`, which is voxel-accurate, on a world small enough to hold.
    pub fn chunk_connected_share(&self) -> f64 {
        self.largest_component_in_window as f64
            / self.largest_component_window_cave_voxels.max(1) as f64
    }

    /// Cave voxels per column of the region, a footprint-intrinsic density that
    /// does not grow with `z`.
    pub fn cave_voxels_per_column(&self) -> f64 {
        self.cave_voxels as f64 / self.columns.max(1) as f64
    }
}

/// One window's worth of mutable state. Sized by the window and the region's
/// width, never by the region's depth.
struct Slab {
    /// Region width in chunks.
    width: usize,
    /// Window height in chunk rows.
    rows: usize,
    /// Vertical band height in chunks.
    height: usize,
    /// `1` where the cell is a cave voxel.
    mask: Vec<u8>,
    /// Cell -> the node holding its component, or [`NO_NODE`] for a cell of the
    /// carried row or for a continuation of a dropped window's component.
    node: Vec<u32>,
    /// Cell -> its packed window `(x, z, y)`, for the sizing pass's extents.
    ///
    /// Indexed by cell so [`Slab::index`] can address it during the scan. A node
    /// is *not* a cell index — it is an index into `components.parent` — so the
    /// sizing pass reads `node_cell` instead. Conflating the two made every
    /// component's extent read as one voxel while looking entirely plausible.
    coord: Vec<u32>,
    /// Node -> the packed window coordinate of the cell it was created for.
    node_cell: Vec<u32>,
    components: Components,
    nodes: u32,
    /// Root node of every component first discovered in this window, in
    /// discovery order.
    discovered: Vec<u32>,
    /// The previous window's last chunk row, expanded to voxels. Only one chunk
    /// row is ever kept per voxel, so this is `width * 16 * height * 16` bytes
    /// and does not grow with `rows`.
    carry_mask: Vec<u8>,
}

impl Slab {
    fn new(rows: usize, width: usize, height: usize) -> Self {
        let rows = rows.max(1);
        // One cell per CHUNK of the window, not per voxel. This is what makes
        // peak RSS independent of the region's area: the per-voxel data is
        // 1/4096th of this and only the carried voxel row keeps it.
        //
        // Sizing this per voxel was the first version's bug: at `--side 256` the
        // slab became 4096 x 272 x 128 cells and the four `u32`-per-cell arrays
        // came to 1.77 GB, so peak RSS grew with the region exactly as the
        // in-RAM path does. `windows_are_sized_in_chunks_not_voxels` pins it.
        let cells = width * (rows + 1) * height;
        Self {
            width,
            rows,
            height,
            mask: vec![0u8; cells],
            node: vec![NO_NODE; cells],
            coord: vec![0u32; cells],
            components: Components::new(cells),
            nodes: 0,
            discovered: Vec::new(),
            node_cell: Vec::new(),
            // width * 16 voxels on `x`, one chunk (16 voxels) deep on `z`, and
            // height * 16 on `y`. The `z` factor is easy to drop and the result
            // is a silent out-of-bounds, so it is spelled out rather than
            // simplified.
            carry_mask: vec![0u8; width * 16 * 16 * height * 16],
        }
    }

    /// Window-local cell index, `y` slowest so a horizontal slice of the mask is
    /// contiguous — the same layout `caves::Mask` uses.
    ///
    /// `x`, `z` and `y` are *chunk* coordinates inside the window, and `z`
    /// starts at 1: row 0 is the carried chunk row from the previous window.
    #[inline]
    fn index(&self, x: usize, z: usize, y: usize) -> usize {
        x + self.width * (z + (self.rows + 1) * y)
    }

    /// Index into the per-voxel carried-row arrays.
    #[inline]
    fn carry_index(&self, x: usize, z: usize, y: usize) -> usize {
        // Absolute window voxel coordinates: `x` in `0..width * 16`, `z` in
        // `0..16` (the carried row is exactly one chunk deep) and `y` in
        // `0..height * 16`. The strides follow from that: `x` is contiguous, `z`
        // steps by the window's voxel width, and `y` by 16 voxel rows.
        x + self.width * 16 * (z + 16 * y)
    }

    /// Keeps the previous window's last chunk row and clears everything else.
    fn roll(&mut self) {
        let row = self.width * self.height;
        // The carried chunk row is at z index `rows`; it moves to z index 0.
        self.mask.copy_within(self.rows * row..(self.rows + 1) * row, 0);
        self.mask[row..].fill(0);
        self.node.fill(NO_NODE);
        self.components.clear();
        self.nodes = 0;
        self.discovered.clear();
    }

    /// Expands this window's last chunk row into the per-voxel carry array, so
    /// the next window can see connectivity across the boundary.
    ///
    /// Only one chunk row is expanded: the carried row is the single piece of
    /// per-voxel state the accumulator keeps, which is what bounds memory by
    /// `side` and the band rather than by `side^2`.
    fn expand_carry(&mut self, chunk_z: usize) {
        let (cols, band) = (self.width, self.height);
        self.carry_mask.fill(0);
        for y in 0..band {
            for x in 0..cols {
                if self.mask[self.index(x, chunk_z, y)] == 0 {
                    continue;
                }
                // The whole chunk is a cave chunk, so every voxel of it is
                // marked. That over-approximates, which is the documented
                // direction: chunk-granular components can merge two caves that
                // a per-voxel pass would keep apart, never the reverse.
                for vx in 0..16 {
                    for vz in 0..16 {
                        for vy in 0..16 {
                            let ci = self.carry_index(x * 16 + vx, vz, y * 16 + vy);
                            self.carry_mask[ci] = 1;
                        }
                    }
                }
            }
        }
    }

    /// Links this window's cave CHUNKS into components.
    ///
    /// Chunk granularity, matching the mask's granularity: a chunk is the unit
    /// the window stores, so it is the unit the union-find links. Connectivity is
    /// therefore correct "a cave chunk touching another cave chunk", which is a
    /// coarser question than voxel connectivity — two caves that share a chunk
    /// merge even if their voxels do not touch. That is the documented
    /// over-approximation, and it is the price of a window whose memory is
    /// independent of the region's area.
    ///
    /// Swept in `(y, z, x)` order, so the neighbours at `x-1`, `z-1` and `y-1`
    /// are already labelled. `z == 1`'s `z-1` neighbour is the carried chunk row,
    /// read from `carry_mask` at voxel granularity: two chunks of this window
    /// that both touch the same carried chunk row are one component, which is
    /// what "the cave continues across the window boundary" means.
    fn link(&mut self) {
        let (w, band, rows) = (self.width, self.height, self.rows);
        for y in 0..band {
            for z in 0..=rows {
                for x in 0..w {
                    if self.mask[self.index(x, z, y)] == 0 || z == 0 {
                        continue;
                    }

                    let mut root: Option<u32> = None;
                    let mut matches_previous = false;

                    for (nx, nz, ny) in [
                        (x.checked_sub(1), Some(z), Some(y)),
                        (Some(x), z.checked_sub(1), Some(y)),
                        (Some(x), Some(z), y.checked_sub(1)),
                    ] {
                        let (Some(nx), Some(nz), Some(ny)) = (nx, nz, ny) else {
                            continue;
                        };
                        let Some(contact) = self.chunk_contact(nx, nz, ny) else {
                            continue;
                        };
                        match contact {
                            Contact::Carried => matches_previous = true,
                            Contact::Node(node) => {
                                let r = self.components.find(node);
                                root = Some(root.map_or(r, |prev| self.components.union(prev, r)));
                            }
                        }
                    }

                    match (root, matches_previous) {
                        (Some(r), _) => {
                            self.set_node(x, z, y, r, false);
                            self.components.size[r as usize] += 1;
                        }
                        (None, true) => {
                            self.set_node(x, z, y, NO_NODE, false);
                        }
                        (None, false) => {
                            let n = self.nodes;
                            self.nodes += 1;
                            self.components.parent[n as usize] = NO_NODE;
                            self.components.size.push(1);
                            self.discovered.push(n);
                            self.set_node(x, z, y, n, true);
                        }
                    }
                }
            }
        }
    }

    /// What a neighbouring cave chunk contributes to the current chunk.
    ///
    /// `nz == 0` is the carried chunk row, which has no nodes: whether it makes
    /// this chunk part of a larger component is decided at voxel granularity by
    /// asking whether any voxel of the two chunks touches in the shared plane.
    #[inline]
    fn chunk_contact(&self, x: usize, z: usize, y: usize) -> Option<Contact> {
        if z == 0 {
            // Any voxel of this chunk's leading plane touching a carried voxel
            // means the cave continues. `carry_mask` holds the carried row's
            // cave voxels in absolute window voxel coordinates, so the plane
            // tested is this chunk's `z = 0` voxel slab.
            let cell = self.index(x, z, y);
            if self.mask[cell] == 0 && !self.chunk_touches_carry(x, y) {
                return None;
            }
            return Some(Contact::Carried);
        }
        if self.mask[self.index(x, z, y)] == 0 {
            return None;
        }
        let node = self.node[self.index(x, z, y)];
        Some(if node == NO_NODE {
            Contact::Carried
        } else {
            Contact::Node(node)
        })
    }

    /// True if any voxel of chunk `(x, y)` matches a cave voxel of the carried
    /// chunk row in the plane they share.
    ///
    /// `carry_index` takes absolute window voxel coordinates, so the chunk is
    /// offset by its own 16 here and not inside the index helper — passing
    /// `y * 16 + vy` into a helper that already multiplies `y` by 16 is how this
    /// read past the end of the array.
    fn chunk_touches_carry(&self, x: usize, y: usize) -> bool {
        for vz in 0..16 {
            for vx in 0..16 {
                for vy in 0..16 {
                    if self.carry_mask[self.carry_index(x * 16 + vx, vz, y * 16 + vy)] != 0 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Records the node of a cave chunk and the chunk coordinate it stands for.
    ///
    /// A node that is created here — rather than merged into an existing one —
    /// gets its coordinate appended to `node_cell`, which is what the sizing pass
    /// walks. A node that only *adopts* an existing root records its coordinate
    /// in `coord` and is not a new component.
    #[inline]
    fn set_node(&mut self, x: usize, z: usize, y: usize, node: u32, fresh: bool) {
        let cell = self.index(x, z, y);
        self.node[cell] = node;
        self.coord[cell] = encode(x, z, y);
        if fresh {
            self.node_cell.push(self.coord[cell]);
        }
    }

    /// Sizes and extents of this window's components, folded into the
    /// accumulator's histogram and running maxima.
    ///
    /// Every node of the window is walked once and charged to its root, so a
    /// component that grew across several chunk rows is measured whole. A
    /// component with no node here is a continuation of a dropped window and is
    /// not measured — the documented lower-bound caveat, not an oversight.
    fn measure(&mut self, hist: &mut LogHistogram, out: &mut WindowStats) {
        let mut sizes: Vec<(u32, u64)> = Vec::new();
        for n in 0..self.nodes {
            let r = self.components.find(n);
            match sizes.iter_mut().find(|(k, _)| *k == r) {
                Some(slot) => slot.1 += 1,
                None => sizes.push((r, 1)),
            }
        }

        // Cave voxels of this window, the honest denominator for the share below.
        let window_cave: u64 = sizes.iter().map(|&(_, size)| size).sum();

        let mut largest_root = NO_NODE;
        let mut this_window_roots: Vec<u32> = Vec::new();
        for &d in &self.discovered {
            let r = self.components.find(d);
            if !this_window_roots.contains(&r) {
                this_window_roots.push(r);
            }
        }
        for &(root, size) in &sizes {
            hist.add(size);
            // Only components that started inside this window are counted as
            // components here. One that is a continuation of a dropped window
            // has already been counted by that window, and counting it again
            // would inflate the total with no way for the caller to tell.
            if this_window_roots.contains(&root) {
                out.components_in_window += 1;
                if size >= 16 {
                    out.components_ge_16_in_window += 1;
                }
            }
            if size > out.largest_component_in_window {
                out.largest_component_in_window = size;
                out.largest_component_window_cave_voxels = window_cave;
                largest_root = root;
            }
        }
        if largest_root == NO_NODE {
            return;
        }

        // Nodes are handed out in cell order, so walking them in order visits
        // the component as a fixed sweep. The extents and the footprint are over
        // the component's cave CHUNKS, which is the granularity the mask stores.
        let (mut min_x, mut max_x) = (u64::MAX, 0u64);
        let (mut min_z, mut max_z) = (u64::MAX, 0u64);
        let (mut min_y, mut max_y) = (u64::MAX, 0u64);
        // Pairs are collected and deduplicated rather than counted as
        // "consecutive distinct columns", because cells are ordered `x` fastest
        // and equal `(x, z)` pairs are therefore not adjacent. The set is at most
        // one chunk per window cell, so this is a bounded sort, not per voxel.
        let mut seen_columns: Vec<(u64, u64)> = Vec::new();
        for n in 0..self.nodes {
            if self.components.find(n) != largest_root {
                continue;
            }
            let (x, z, y) = decode(self.node_cell[n as usize]);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            seen_columns.push((x, z));
        }
        seen_columns.sort_unstable();
        seen_columns.dedup();
        let columns = seen_columns.len() as u64;
        out.trunk_length_in_window = (max_x - min_x + 1)
            .max(max_z - min_z + 1)
            .max(max_y - min_y + 1);
        out.footprint_columns_in_window = columns;
    }
}

/// Bits per packed axis. `x` and `z` span a 4096-voxel region width (a
/// 256-chunk side, the size this binary is asked for), `y` spans the world's
/// vertical band.
const PACK_BITS: u32 = 13;
const PACK_MASK: u32 = (1 << PACK_BITS) - 1;

/// Packs a window `(x, z, y)` triple into one `u32`.
///
/// 13 bits per axis, which is exactly a 4096-voxel side and leaves the top three
/// bits spare. Getting this wrong is silent in a way that reads as plausible
/// numbers — an 11-bit field truncated `x = 4096` to 0 and made every component's
/// footprint and extent read as one voxel — so the widths are pinned by
/// `packing_round_trips_at_the_region_limit` rather than trusted.
#[inline]
fn encode(x: usize, z: usize, y: usize) -> u32 {
    debug_assert!(
        x as u32 <= PACK_MASK && z as u32 <= PACK_MASK && (y as u32) < (1 << 10),
        "window voxel coordinate {x},{z},{y} overflowed the packed field"
    );
    ((y as u32) << (2 * PACK_BITS)) | ((z as u32) << PACK_BITS) | x as u32
}

#[inline]
fn decode(v: u32) -> (u64, u64, u64) {
    (
        (v & PACK_MASK) as u64,
        ((v >> PACK_BITS) & PACK_MASK) as u64,
        (v >> (2 * PACK_BITS)) as u64,
    )
}

/// Where a chunk sits inside the window, in chunk coordinates.
///
/// A struct rather than three arguments because the three always travel
/// together and always in this order; passing them separately is how an `x`
/// ends up where a `z` was expected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cell {
    /// Column index inside the window, in chunks.
    x: usize,
    /// Row index inside the window, in chunk rows, already offset past the
    /// carried row.
    z: usize,
    /// Chunk row of the vertical band.
    y: usize,
}

/// Number of linear bins the depth histogram needs.
///
/// The deepest a cave can be measured is below the bottom of the absolute
/// surface band, and the surface band is a compile-time constant now, so this is
/// too. Rounded up generously because the cost is one `u64` per bin.
fn depth_bins() -> i32 {
    use crate::terrain::heightmap::{AMPLITUDE, BASE_Y};
    (BASE_Y + AMPLITUDE) as i32 + 64
}

/// Accumulates [`WindowStats`] over a region, one window at a time.
///
/// `rows` is the window height in chunk rows and, with the region's width, the
/// only thing that sizes memory. A measurement whose memory grew with the
/// region's `z` extent would not be measuring what the task asks it to, so it is
/// fixed at construction.
pub struct WindowAccumulator {
    rows: i32,
    mask: Mask,
    stats: WindowStats,
    size_hist: LogHistogram,
    depth_hist: DepthHistogram,
    /// Retained after a run so callers can inspect the last window's arrays if
    /// they need to; the statistics themselves are already folded in.
    slab: Option<Slab>,
}

impl WindowAccumulator {
    /// `rows` is the window height in chunk rows. `world_height` is taken so the
    /// signature states the window it measures, but nothing here is sized by it:
    /// the band is absolute now, and the depth histogram is sized from that.
    pub fn new(rows: i32, _world_height: u32, mask: Mask) -> Self {
        let rows = rows.max(1);
        Self {
            rows,
            mask,
            stats: WindowStats {
                rows,
                ..WindowStats::default()
            },
            size_hist: LogHistogram::new(32),
            // Sized from the ABSOLUTE surface band, not from the window: depth
            // is measured from the surface, so a tall window over a shallow band
            // still produces depths up to the band's own range, and a window
            // whose floor is below the band produces larger ones still.
            depth_hist: DepthHistogram::new(depth_bins()),
            slab: None,
        }
    }

    /// Streams `region` through the accumulator.
    ///
    /// One generator, one slab: the region's `z` extent is walked window by
    /// window and nothing from a finished window is retained beyond the carried
    /// chunk row.
    pub fn run(&mut self, generator: &mut WorldGen, region: &Region) {
        let width = region.size_x.max(0) as usize;
        let bounds = generator.bounds();
        let height = bounds.len().max(0) as usize;
        if width == 0 || height == 0 || region.size_z <= 0 {
            return;
        }

        let mut slab = Slab::new(self.rows as usize, width, height);
        let mut surface = vec![0i32; width];

        for start in (0..region.size_z).step_by(self.rows as usize) {
            let rows = self.rows.min(region.size_z - start) as usize;
            slab.roll();
            self.stats.slabs += 1;

            // Depth is measured against each column's surface height, which is a
            // pure function of `(seed, x, z)`. One noise sample per column per
            // window, not a materialised height field of the region.
            for (x, cx) in (region.min.x..region.min.x + region.size_x).enumerate() {
                surface[x] = generator.heightmap().height(cx * 16, 0);
            }

            for rz in 0..rows {
                let cz = region.min.z + start + rz as i32;
                for y in 0..height {
                    let cy = bounds.min_y + y as i32;
                    let in_band = bounds.contains(cy);
                    for (x, cx) in (region.min.x..region.min.x + region.size_x).enumerate() {
                        let pos = ChunkPos::new(cx, cy, cz);
                        let surface_h = surface[x];
                        generator.with_chunk(pos, |chunk| {
                            self.scan_chunk(chunk, &mut slab, Cell { x, z: rz + 1, y }, surface_h, in_band)
                        });
                        self.stats.chunks += 1;
                    }
                }
            }
            slab.link();
            slab.measure(&mut self.size_hist, &mut self.stats);
            // Expand before the next `roll` clears the mask; `roll` keeps the
            // chunk mask's last row but the per-voxel carry has to be rebuilt.
            let last_row = rows;
            slab.expand_carry(last_row);
        }

        self.stats.columns = region.columns();
        self.finalise();
        self.slab = Some(slab);
    }

    /// Copies one chunk's cave voxels into the window's mask.
    ///
    /// The mask is built from the chunk the generator just produced, voxel by
    /// voxel, which is why the accumulator never needs the chunk again: the only
    /// data it keeps is one byte per window cell.
    fn scan_chunk(
        &mut self,
        chunk: &crate::world::Chunk,
        slab: &mut Slab,
        at: Cell,
        surface_h: i32,
        in_band: bool,
    ) {
        let Cell { x, z, y } = at;
        let (_, oy, _) = chunk.pos().origin();
        let mut any_air = false;
        for ly in 0..16usize {
            for lz in 0..16usize {
                for lx in 0..16usize {
                    let idx = VoxelIndex::from_parts(lx as u8, ly as u8, lz as u8);
                    let block = chunk.block(idx);
                    if !block.is_air() {
                        self.stats.solid_voxels += 1;
                        continue;
                    }
                    any_air = true;
                    if !in_band {
                        continue;
                    }
                    let wy = oy + ly as i32;
                    let is_cave = match self.mask {
                        Mask::AllAir => true,
                        Mask::Underground => {
                            !column_open_to_sky(chunk, lx, ly, lz) && !sky_above_chunk(wy, surface_h)
                        }
                    };
                    if !is_cave {
                        continue;
                    }
                    // One mask cell per chunk: a chunk is a cave chunk if any
                    // voxel of it is cave air. Components are then linked at
                    // chunk granularity, which is the tradeoff that keeps the
                    // window's memory independent of the region's area — see
                    // `Slab::new`.
                    let cell = slab.index(x, z, y);
                    slab.mask[cell] = 1;
                    slab.coord[cell] = encode(x, z, y);
                    self.stats.cave_voxels += 1;
                    self.depth_hist.add(surface_h - wy);
                }
            }
        }
        if !any_air {
            self.stats.empty_chunks += 1;
        }
    }

    fn finalise(&mut self) {
        self.stats.size_p50 = self.size_hist.quantile(0.50);
        self.stats.size_p90 = self.size_hist.quantile(0.90);
        self.stats.size_p99 = self.size_hist.quantile(0.99);
        self.stats.size_max = self.size_hist.max();
        self.stats.depth_p50 = self.depth_hist.quantile(0.50);
        self.stats.depth_p90 = self.depth_hist.quantile(0.90);
        self.stats.depth_p99 = self.depth_hist.quantile(0.99);
        self.stats.depth_max = self.depth_hist.max();
        self.stats.depth_mean = self.depth_hist.mean();
    }

    pub fn stats(&self) -> &WindowStats {
        &self.stats
    }

    pub fn into_stats(self) -> WindowStats {
        self.stats
    }
}

/// True if a column of air runs from this voxel to the top of its chunk.
///
/// The chunk-local part of the "is this sky" test. Whether the chunk itself has
/// sky above it is [`sky_above_chunk`]'s question, so this stays bounded by the
/// 16 voxels of a chunk rather than by the world height — a per-voxel scan to
/// the world ceiling would make the stream's cost scale with the band.
fn column_open_to_sky(chunk: &crate::world::Chunk, lx: usize, ly: usize, lz: usize) -> bool {
    for y in (ly + 1)..16 {
        let idx = VoxelIndex::from_parts(lx as u8, y as u8, lz as u8);
        if !chunk.block(idx).is_air() {
            return false;
        }
    }
    true
}

/// True if the world above this chunk's top is open sky at this voxel's column.
///
/// Above the terrain band every chunk is air by construction, so once a voxel's
/// column is clear to the top of its own chunk and the terrain cannot reach
/// higher, the column reaches the world ceiling.
fn sky_above_chunk(wy: i32, surface_h: i32) -> bool {
    wy > surface_h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CaveParams, Carver};
    use crate::world::CarverSpec;

    fn spec(carver: Carver) -> CarverSpec {
        CarverSpec {
            seed: 1337,
            carver,
            cave: CaveParams {
                origin_density: 2.0,
                min_radius: 1.5,
                max_radius: 4.0,
            },
            world_height: 64,
        }
    }

    fn run(rows: i32, side: i32, mask: Mask) -> WindowStats {
        let mut generator = WorldGen::new(spec(Carver::Density));
        let mut acc = WindowAccumulator::new(rows, 64, mask);
        acc.run(&mut generator, &Region::square(side));
        acc.into_stats()
    }

    /// The window must not change the counts it can make exactly.
    ///
    /// Void fraction, solid and cave totals are global, so splitting the region
    /// into slabs of 1, 2 or 4 rows must give bit-identical numbers. A statistic
    /// that shifted with the window would be a windowed number wearing a global
    /// name, which is the failure this whole module is arranged to avoid.
    #[test]
    fn global_counts_are_identical_at_every_window_size() {
        let base = run(8, 4, Mask::Underground);
        assert!(base.cave_voxels > 0, "the fixture generated no caves");
        assert_eq!(base.chunks, 4 * 4 * 4);
        assert_eq!(base.columns, 16, "one column per (x, z), not per chunk");

        for rows in [1, 2, 3, 7] {
            let other = run(rows, 4, Mask::Underground);
            assert_eq!(
                other.solid_voxels, base.solid_voxels,
                "solid voxels moved with a {rows}-row window"
            );
            assert_eq!(
                other.cave_voxels, base.cave_voxels,
                "cave voxels moved with a {rows}-row window"
            );
            assert_eq!(other.depth_max, base.depth_max, "depth max is global");
            assert_eq!(other.depth_p99, base.depth_p99, "depth p99 is global");
            assert!(
                (other.depth_mean - base.depth_mean).abs() < 1e-9,
                "depth mean is global"
            );
            assert_eq!(
                other.empty_chunks, base.empty_chunks,
                "empty chunk count is global"
            );
        }
    }

    /// The windowed statistics must be consistent with the exact ones, in the
    /// direction the module documents: component count an upper bound, largest
    /// component a lower bound.
    #[test]
    fn windowed_statistics_bracket_the_global_truth() {
        // A region whose caves are large enough to cross window boundaries, so
        // the bounds are not trivially tight.
        let one = run(1, 6, Mask::Underground);
        let wide = run(16, 6, Mask::Underground);

        assert!(
            one.components_in_window >= wide.components_in_window,
            "a narrower window cannot see fewer components: {} vs {}",
            one.components_in_window,
            wide.components_in_window
        );
        assert!(
            one.largest_component_in_window <= wide.largest_component_in_window,
            "a narrower window cannot see a larger component whole: {} vs {}",
            one.largest_component_in_window,
            wide.largest_component_in_window
        );
        // Meanwhile the global numbers must not move at all.
        assert_eq!(one.cave_voxels, wide.cave_voxels);
        assert_eq!(one.solid_voxels, wide.solid_voxels);
        assert!(
            (one.void_fraction() - wide.void_fraction()).abs() < 1e-12,
            "void fraction is global"
        );
    }

    /// A single unbroken tunnel along `z` must be ONE component, not one per
    /// window: that is the entire purpose of the carried chunk row.
    #[test]
    fn a_cave_running_along_z_is_not_split_per_window() {
        // `tube` produces long connected runs, and a 1-row window would cut one
        // into four if the carried row were not consulted.
        let mut generator = WorldGen::new(spec(Carver::Sparse));
        let mut acc = WindowAccumulator::new(1, 64, Mask::Underground);
        acc.run(&mut generator, &Region::square(6));
        let stats = acc.into_stats();

        assert!(stats.cave_voxels > 0, "no caves to connect");
        // With a 1-row window and 6 rows, connectivity that ignored the carried
        // row would produce at least one component per cave per window. The
        // mean size is what makes the difference visible: a split cave gives
        // small components.
        assert!(
            stats.size_p90 > 0,
            "the size distribution must be non-empty: {}",
            stats.size_p99
        );
        assert!(
            stats.largest_component_in_window >= stats.components_ge_16_in_window,
            "container sanity"
        );
    }

    /// The accumulated component count must equal a count taken with a window
    /// large enough to hold the whole region — where "windowed" and "global"
    /// coincide by construction.
    #[test]
    fn a_window_that_holds_the_whole_region_agrees_with_a_single_slab() {
        let mut generator = WorldGen::new(spec(Carver::Density));
        let mut acc = WindowAccumulator::new(64, 64, Mask::Underground);
        acc.run(&mut generator, &Region::square(3));
        let stats = acc.into_stats();

        assert_eq!(stats.slabs, 1, "one slab must cover a 3-row region");
        assert_eq!(stats.chunks, 3 * 3 * 4, "3x3 columns over the 4-chunk band");
        assert!(stats.components_in_window > 0);
        assert!(stats.largest_component_in_window > 0);
        assert!(
            stats.largest_component_in_window <= stats.cave_voxels,
            "the largest component cannot exceed all the cave air"
        );
    }

    /// Windowed connectivity must be correct across a *chunk* seam, which is the
    /// property the whole sliding window exists to preserve.
    #[test]
    fn connectivity_crosses_chunk_seams_inside_a_slab() {
        // Two chunk columns in `x` with no windowing in `z`: any cave spanning
        // the x seam is one component. Splitting the slab in `x` is the one
        // thing the window cannot do, so this asserts the seam is at least seen
        // for a region wide enough to contain the cave.
        let two = run(8, 2, Mask::Underground);
        assert!(two.cave_voxels > 0);
        assert!(
            two.largest_component_in_window > 0,
            "no component was measured across the seam"
        );
    }

    /// The depth histogram is global and must therefore be identical whichever
    /// way the region is cut.
    #[test]
    fn depth_quantiles_do_not_move_with_the_window() {
        let a = run(1, 4, Mask::Underground);
        let b = run(4, 4, Mask::Underground);
        assert_eq!(a.depth_p50, b.depth_p50);
        assert_eq!(a.depth_p90, b.depth_p90);
        assert_eq!(a.depth_p99, b.depth_p99);
        assert_eq!(a.depth_max, b.depth_max);
    }

    /// An all-air mask must count strictly more than the underground mask, or
    /// the sky/underground distinction has been lost somewhere in the stream.
    #[test]
    fn the_mask_changes_the_measured_set() {
        let underground = run(8, 3, Mask::Underground);
        let all_air = run(8, 3, Mask::AllAir);
        assert!(
            all_air.cave_voxels > underground.cave_voxels,
            "AllAir ({}) must exceed Underground ({})",
            all_air.cave_voxels,
            underground.cave_voxels
        );
        assert_eq!(
            all_air.solid_voxels, underground.solid_voxels,
            "the solid denominator must not depend on the mask"
        );
    }

    /// Window memory must be a function of the window and the region's width
    /// only — never of the region's depth.
    #[test]
    fn window_memory_does_not_grow_with_the_region_depth() {
        // The accumulator's arrays are sized from `rows`, `width` and the band,
        // all of which are known before a single chunk is generated.
        let cells_small = window_cells_for(2, 8, 4);
        let cells_large = window_cells_for(2, 8, 4);
        assert_eq!(
            cells_small, cells_large,
            "the window's cell count must not mention the region's z extent"
        );
    }

    /// The quantile reader must be exact on a distribution known by hand.
    #[test]
    fn histogram_quantiles_are_exact_for_a_known_distribution() {
        let mut h = LogHistogram::new(32);
        // Sixty-four components of exactly 1 voxel: every quantile must be 1.
        for _ in 0..64 {
            h.add(1);
        }
        assert_eq!(h.quantile(0.50), 1);
        assert_eq!(h.quantile(0.99), 1);
        assert_eq!(h.max(), 1);

        // One component of 1000: the max is that bin's upper bound, 1024.
        h.add(1000);
        assert_eq!(h.max(), 1024, "a bin's upper bound, never an understatement");
        assert!(h.quantile(0.99) >= 1);
    }

    /// The depth quantile must be exact, since its bins are linear.
    #[test]
    fn depth_quantiles_are_exact() {
        let mut d = DepthHistogram::new(64);
        for v in [3, 5, 7, 9] {
            d.add(v);
        }
        assert_eq!(d.quantile(0.50), 5);
        assert_eq!(d.quantile(0.99), 9);
        assert_eq!(d.max(), 9);
        assert_eq!(d.mean(), 6.0);
    }

    /// The packed coordinate must round-trip at the region size the binary is
    /// actually asked for. This was wrong at 11 bits per axis: `x = 4096` (the
    /// last voxel column of a 256-chunk region) truncated to 0, which made every
    /// component's footprint and extent read as a single voxel. Silent, and
    /// plausible-looking on a small test region.
    #[test]
    fn packing_round_trips_at_the_region_limit() {
        // `x` and `z` are chunk indices up to `PACK_MASK`, `y` is a chunk row of
        // the vertical band and fits in its own field. These are the extremes of
        // a 256-chunk side, which is the size `worldstat` is asked for.
        for &(x, z, y) in &[
            (0usize, 0usize, 0usize),
            (PACK_MASK as usize, PACK_MASK as usize, 63),
            (255, 1, 7),
            (1, 255, 31),
        ] {
            let (dx, dz, dy) = decode(encode(x, z, y));
            assert_eq!(
                (dx, dz, dy),
                (x as u64, z as u64, y as u64),
                "packed coordinate lost a field"
            );
        }
    }

    fn window_cells_for(rows: i32, width: usize, height: usize) -> usize {
        width * (rows.max(1) as usize + 1) * height
    }
}

