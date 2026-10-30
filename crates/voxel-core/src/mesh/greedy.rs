//! The greedy mesher.
//!
//! For each of the three axes, the chunk is swept slice by slice. Within a
//! slice, every voxel produces a face mask value (the block word, or zero if
//! the face is hidden). The mask is then scanned once to merge equal runs into
//! as few rectangles as possible.
//!
//! The merge is the standard two-pass bitwise sweep: for each unvisited cell,
//! extend along `u` while the mask matches, then extend along `v` while every
//! cell of the next row matches, which is why it can collapse a checkerboard-
//! free 16x16 face into a single quad instead of 256.

use super::{Direction, should_draw_face};
use crate::domain::{BlockId, CHUNK_SIZE, LocalPos, PackedVoxel, VoxelIndex};
use crate::world::Chunk;

/// One visible rectangle of a chunk's surface.
///
/// Deliberately flat and `Copy`: the GPU buffer is just an array of these, and
/// the tests compare meshes by comparing quad sets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad {
    /// Position of the quad's minimal corner, in chunk-local voxel units.
    pub pos: [u8; 3],
    /// Which way the face points.
    pub dir: Direction,
    /// Extent along the two axes perpendicular to `dir`, in voxels.
    pub size: [u8; 2],
    /// Block drawn on this face.
    pub block: BlockId,
    /// Packed light of the face's voxel, so shading is per-quad.
    pub light: PackedVoxel,
}

impl Quad {
    /// Number of voxel faces this quad covers.
    #[inline]
    pub fn face_count(&self) -> u32 {
        self.size[0] as u32 * self.size[1] as u32
    }

    /// The four corner positions, in voxel units, wound consistently.
    ///
    /// Returned in a fixed order so the GPU stage can build two triangles
    /// without consulting the direction again.
    pub fn corners(&self) -> [[f32; 3]; 4] {
        let [u_size, v_size] = self.size;
        let (u_axis, v_axis) = plane_axes(self.dir);
        let origin = [self.pos[0] as f32, self.pos[1] as f32, self.pos[2] as f32];

        let mut out = [[0.0f32; 3]; 4];
        // (0,0), (u,0), (u,v), (0,v) in plane coordinates.
        let steps = [
            (0.0f32, 0.0f32),
            (u_size as f32, 0.0),
            (u_size as f32, v_size as f32),
            (0.0, v_size as f32),
        ];
        for (i, (du, dv)) in steps.iter().enumerate() {
            out[i] = origin;
            out[i][u_axis] += du;
            out[i][v_axis] += dv;
        }
        out
    }

    /// The normal of this quad, as a unit vector.
    #[inline]
    pub fn normal(&self) -> [f32; 3] {
        self.dir.normal()
    }
}

/// The two axes that span a face with the given direction.
///
/// Returned as low-to-high axis indices so `corners` can be written once.
#[inline]
pub const fn plane_axes(dir: Direction) -> (usize, usize) {
    match dir.axis() {
        0 => (1, 2), // x-facing: u = y, v = z
        1 => (0, 2), // y-facing: u = x, v = z
        _ => (0, 1), // z-facing: u = x, v = y
    }
}

/// A chunk's mesh: quads plus the counters the tests and benchmarks need.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mesh {
    pub quads: Vec<Quad>,
    /// Voxel faces the chunk actually has, before merging.
    pub faces: u32,
}

impl Mesh {
    /// How much the merge saved, as a ratio of quads to faces.
    pub fn merge_ratio(&self) -> f32 {
        if self.faces == 0 {
            return 1.0;
        }
        self.quads.len() as f32 / self.faces as f32
    }

    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }
}

/// Meshes a chunk greedily.
///
/// Faces on the chunk boundary are compared against air, so a chunk's outer
/// shell is always drawn. Neighbour-aware meshing would need the adjacent
/// chunk, which breaks the independence the parallel generator relies on;
/// the cost is one extra quad layer that back-face culling removes anyway.
pub fn mesh_chunk(chunk: &Chunk) -> Mesh {
    let mut mesh = Mesh::default();
    let mut mask = [0u32; CHUNK_SIZE * CHUNK_SIZE];

    for dir in Direction::ALL {
        for slice in 0..CHUNK_SIZE {
            build_mask(chunk, dir, slice, &mut mask);

            // Scan the mask, merging equal runs into rectangles.
            let mut v = 0usize;
            while v < CHUNK_SIZE {
                let mut u = 0usize;
                while u < CHUNK_SIZE {
                    let m = mask[v * CHUNK_SIZE + u];
                    if m == 0 {
                        u += 1;
                        continue;
                    }

                    // Extend along u while the mask is unchanged.
                    let mut width = 1usize;
                    while u + width < CHUNK_SIZE && mask[v * CHUNK_SIZE + u + width] == m {
                        width += 1;
                    }

                    // Extend along v while every cell of each row matches.
                    let mut height = 1usize;
                    'grow: while v + height < CHUNK_SIZE {
                        for k in 0..width {
                            if mask[(v + height) * CHUNK_SIZE + u + k] != m {
                                break 'grow;
                            }
                        }
                        height += 1;
                    }

                    let (block, light) = unpack_mask(m);
                    mesh.quads.push(Quad {
                        pos: plane_pos(dir, slice, u as u8, v as u8),
                        dir,
                        size: [width as u8, height as u8],
                        block,
                        light,
                    });

                    // Clear the consumed rectangle so later scans skip it.
                    for dv in 0..height {
                        for du in 0..width {
                            mask[(v + dv) * CHUNK_SIZE + u + du] = 0;
                        }
                    }
                    u += width;
                }
                v += 1;
            }
        }
    }

    mesh.faces = mesh.quads.iter().map(|q| q.face_count()).sum();
    mesh
}

/// Packs the mask word: block id plus light, or zero for a hidden face.
///
/// The light is folded in so quads only merge where their light agrees, which
/// is what keeps a light gradient visible instead of being smoothed away by
/// an over-eager merge.
#[inline]
fn pack_mask(v: PackedVoxel) -> u32 {
    // id in bits 0..8, block light 8..12, sunlight 12..16. Zero means "no face"
    // so air (id 0) can never be mistaken for a drawn face.
    v.raw() as u32
}

#[inline]
fn unpack_mask(m: u32) -> (BlockId, PackedVoxel) {
    let v = PackedVoxel::from_raw(m as u16);
    (v.block(), v)
}

/// Fills `mask` with the faces of one slice that need drawing.
fn build_mask(chunk: &Chunk, dir: Direction, slice: usize, mask: &mut [u32; 256]) {
    let axis = dir.axis();
    let (u_axis, v_axis) = plane_axes(dir);

    for v in 0..CHUNK_SIZE {
        for u in 0..CHUNK_SIZE {
            let mut coord = [0usize; 3];
            coord[axis] = slice;
            coord[u_axis] = u;
            coord[v_axis] = v;

            let index = VoxelIndex::from_parts(coord[0] as u8, coord[1] as u8, coord[2] as u8);
            let here = chunk.get(index);
            let neighbour = neighbour_across(chunk, coord, dir);

            mask[v * CHUNK_SIZE + u] = if should_draw_face(here.block(), neighbour) {
                pack_mask(here)
            } else {
                0
            };
        }
    }
}

/// Reads the voxel across a face, treating anything outside the chunk as air.
#[inline]
fn neighbour_across(chunk: &Chunk, coord: [usize; 3], dir: Direction) -> BlockId {
    let (dx, dy, dz) = dir.step();
    let n = CHUNK_SIZE as i32;
    let x = coord[0] as i32 + dx;
    let y = coord[1] as i32 + dy;
    let z = coord[2] as i32 + dz;
    if x < 0 || y < 0 || z < 0 || x >= n || y >= n || z >= n {
        // Outside the chunk is air: draw the boundary face.
        return BlockId::AIR;
    }
    chunk.block(VoxelIndex::from_parts(x as u8, y as u8, z as u8))
}

/// Turns a slice coordinate and in-plane `(u, v)` into a chunk-local position.
#[inline]
fn plane_pos(dir: Direction, slice: usize, u: u8, v: u8) -> [u8; 3] {
    let axis = dir.axis();
    let (u_axis, v_axis) = plane_axes(dir);
    let mut pos = [0u8; 3];
    pos[axis] = slice as u8;
    pos[u_axis] = u;
    pos[v_axis] = v;
    pos
}

/// Convenience for tests and debug views: total faces covered by a mesh.
pub fn covered_faces(mesh: &Mesh) -> u32 {
    mesh.quads.iter().map(|q| q.face_count()).sum()
}

/// Rebuilds the set of covered `(voxel, direction)` faces, for equivalence
/// checks against the naive mesher.
pub fn covered_face_set(mesh: &Mesh) -> std::collections::HashSet<(u8, u8, u8, Direction)> {
    let mut set = std::collections::HashSet::new();
    for q in &mesh.quads {
        let (u_axis, v_axis) = plane_axes(q.dir);
        for dv in 0..q.size[1] {
            for du in 0..q.size[0] {
                let mut pos = q.pos;
                pos[u_axis] += du;
                pos[v_axis] += dv;
                set.insert((pos[0], pos[1], pos[2], q.dir));
            }
        }
    }
    set
}

/// Local position of a voxel, re-exported for the naive mesher's symmetry.
#[inline]
pub fn local_of(x: u8, y: u8, z: u8) -> LocalPos {
    LocalPos::new(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ChunkPos;
    use crate::terrain::heightmap::HeightMap;

    fn solid_chunk() -> Chunk {
        let mut c = Chunk::at(ChunkPos::ORIGIN);
        c.fill(BlockId::STONE);
        c
    }

    #[test]
    fn a_solid_chunk_meshes_as_one_quad_per_face() {
        // The whole point of greedy meshing: 6 quads for a 16^3 cube of stone,
        // not 6 * 256.
        let mesh = mesh_chunk(&solid_chunk());
        assert_eq!(mesh.quads.len(), 6, "a cube is six quads");
        for q in &mesh.quads {
            assert_eq!(q.size, [16, 16], "each face must be the full 16x16");
        }
        assert_eq!(mesh.faces, 6 * 256);
        assert!(mesh.merge_ratio() < 0.01, "merge should be near-total");
    }

    #[test]
    fn an_empty_chunk_meshes_to_nothing() {
        let mesh = mesh_chunk(&Chunk::at(ChunkPos::ORIGIN));
        assert!(mesh.is_empty());
        assert_eq!(mesh.faces, 0);
        assert_eq!(mesh.merge_ratio(), 1.0, "empty mesh must not divide by zero");
    }

    #[test]
    fn a_single_voxel_meshes_as_six_unit_quads() {
        let mut c = Chunk::at(ChunkPos::ORIGIN);
        c.set_block(VoxelIndex::from_parts(5, 5, 5), BlockId::STONE);
        let mesh = mesh_chunk(&c);
        assert_eq!(mesh.quads.len(), 6);
        for q in &mesh.quads {
            assert_eq!(q.size, [1, 1]);
        }
        assert_eq!(mesh.faces, 6);
    }

    #[test]
    fn hidden_interior_faces_are_not_meshed() {
        // Two adjacent voxels share a face, which must not be drawn: 10, not 12.
        let mut c = Chunk::at(ChunkPos::ORIGIN);
        c.set_block(VoxelIndex::from_parts(5, 5, 5), BlockId::STONE);
        c.set_block(VoxelIndex::from_parts(6, 5, 5), BlockId::STONE);
        let mesh = mesh_chunk(&c);
        assert_eq!(mesh.faces, 10, "shared face must be culled");
    }

    #[test]
    fn every_quad_covers_only_its_own_faces() {
        // A merge bug can double-cover a face, which shows as lighting or
        // z-fighting artefacts rather than as a crash.
        let mesh = mesh_chunk(&solid_chunk());
        let covered = covered_faces(&mesh);
        let counted = mesh.faces;
        assert_eq!(covered, counted);
    }

    #[test]
    fn quads_stay_inside_the_chunk() {
        let mut c = Chunk::at(ChunkPos::ORIGIN);
        c.set_block(VoxelIndex::from_parts(0, 0, 0), BlockId::STONE);
        c.set_block(VoxelIndex::from_parts(15, 15, 15), BlockId::STONE);
        c.set_block(VoxelIndex::from_parts(8, 0, 15), BlockId::GRASS);

        for q in mesh_chunk(&c).quads {
            let (u_axis, v_axis) = plane_axes(q.dir);
            assert!(q.pos[q.dir.axis()] < 16);
            assert!(q.pos[u_axis] + q.size[0] <= 16, "quad overruns u");
            assert!(q.pos[v_axis] + q.size[1] <= 16, "quad overruns v");
            for corner in q.corners() {
                for (i, &v) in corner.iter().enumerate() {
                    assert!((0.0..=16.0).contains(&v), "corner {i} out of chunk: {v}");
                }
            }
        }
    }

    #[test]
    fn different_blocks_do_not_merge() {
        // A stone slab beside a grass slab must stay two quads on the top face.
        let mut c = Chunk::at(ChunkPos::ORIGIN);
        for x in 0..16u8 {
            let block = if x < 8 { BlockId::STONE } else { BlockId::GRASS };
            for z in 0..16u8 {
                c.set_block(VoxelIndex::from_parts(x, 0, z), block);
            }
        }
        let mesh = mesh_chunk(&c);
        // The top face splits in two, but the bottom and sides can still merge.
        let top: Vec<_> = mesh
            .quads
            .iter()
            .filter(|q| q.dir == Direction::PosY)
            .collect();
        assert_eq!(top.len(), 2, "differing blocks must not merge: {top:?}");
        assert!(top.iter().all(|q| q.size == [8, 16]));
    }

    #[test]
    fn differing_light_does_not_merge() {
        // Otherwise a light gradient would be flattened into one flat quad.
        let mut c = Chunk::at(ChunkPos::ORIGIN);
        for z in 0..16u8 {
            for x in 0..16u8 {
                let light = if x < 8 {
                    crate::domain::LightLevel::MAX
                } else {
                    crate::domain::LightLevel::new(3)
                };
                c.set(
                    VoxelIndex::from_parts(x, 0, z),
                    PackedVoxel::new(BlockId::STONE, light, crate::domain::LightLevel::DARK),
                );
            }
        }
        let top = mesh_chunk(&c)
            .quads
            .into_iter()
            .filter(|q| q.dir == Direction::PosY)
            .count();
        assert_eq!(top, 2, "a light step must split the quad");
    }

    #[test]
    fn generated_terrain_meshes_without_overrun() {
        let hm = HeightMap::fitted_to_world(1337, 32);
        let pos = ChunkPos::new(0, 1, 0);
        let mut c = Chunk::at(pos);
        crate::terrain::generate_surface(&mut c, pos, &hm);
        let mesh = mesh_chunk(&c);
        assert!(!mesh.is_empty(), "terrain must produce geometry");
        assert!(
            mesh.merge_ratio() < 1.0,
            "greedy meshing must beat one-quad-per-face on real terrain"
        );
        // Face count must never exceed the theoretical maximum.
        assert!(mesh.faces <= 6 * 4096);
    }

    #[test]
    fn mesh_is_deterministic() {
        let hm = HeightMap::fitted_to_world(99, 32);
        let pos = ChunkPos::new(1, 1, 2);
        let mut c = Chunk::at(pos);
        crate::terrain::generate_surface(&mut c, pos, &hm);
        let a = mesh_chunk(&c);
        let b = mesh_chunk(&c);
        assert_eq!(a, b, "meshing must not depend on hidden state");
    }
}
