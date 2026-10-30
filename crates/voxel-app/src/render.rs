//! Turning chunk meshes into GPU vertex data.
//!
//! One vertex struct, one interleaved buffer per chunk, no index buffer: greedy
//! quads are already minimal, and two triangles per quad written out directly
//! is less wiring than an index buffer earns back at this scale.
//!
//! Shading is baked into the vertex colour rather than read from a texture,
//! which keeps the POC's dependency list to just `wgpu` and `winit`. The colour
//! combines the block's tint, the face direction and the voxel's light level, so
//! the light pass implemented in `voxel-core` is actually visible on screen.

use bytemuck::{Pod, Zeroable};
use voxel_core::mesh::{Direction, Mesh, Quad};
use voxel_core::{BlockId, ChunkPos};

/// A vertex as the GPU sees it: position, colour, one padding word.
///
/// `repr(C)` plus `Pod` means this can be uploaded without conversion. The
/// struct is padded to 32 bytes so the stride is a multiple of 16, which some
/// backends prefer for vertex buffers.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    /// Chunk-local position, in voxels.
    pub position: [f32; 3],
    /// Linear RGB, already shaded.
    pub color: [f32; 3],
    /// Padding so the stride is 32 rather than 28 bytes, keeping it a multiple
    /// of 16 which some backends require for vertex buffer alignment.
    pub _pad: [f32; 2],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
        ],
    };
}

/// Per-face brightness, replacing a normal vector.
///
/// Hard-coded per direction rather than computed from a light vector: the
/// terrain is stylised, and fixed axis shading makes the block grid readable at
/// a glance. The values are the classic voxel-face ramp.
pub const fn face_shade(dir: Direction) -> f32 {
    match dir {
        Direction::PosY => 1.00, // top, brightest
        Direction::NegY => 0.45, // bottom
        Direction::PosX | Direction::NegX => 0.80,
        Direction::PosZ | Direction::NegZ => 0.62,
    }
}

/// Base tint for a block.
pub const fn block_tint(block: BlockId) -> [f32; 3] {
    match block {
        BlockId::AIR => [0.0, 0.0, 0.0],
        BlockId::STONE => [0.55, 0.55, 0.58],
        BlockId::DIRT => [0.52, 0.37, 0.26],
        BlockId::GRASS => [0.34, 0.61, 0.24],
        BlockId::SAND => [0.87, 0.80, 0.53],
        BlockId::WATER => [0.20, 0.35, 0.78],
        // Magenta by construction: an unknown id must be impossible to miss.
        _ => [1.0, 0.0, 1.0],
    }
}

/// Minimum brightness so a fully dark cave still reads as shape, not void.
pub const AMBIENT: f32 = 0.08;

/// Combines tint, face shade and light into the vertex colour.
pub fn shade(block: BlockId, dir: Direction, light: u8) -> [f32; 3] {
    let tint = block_tint(block);
    let light = (light as f32 / 15.0).clamp(0.0, 1.0);
    // Light 15 keeps the face at full shade; darker voxels fall toward ambient.
    let level = AMBIENT + (1.0 - AMBIENT) * light;
    let brightness = face_shade(dir) * level;
    [
        tint[0] * brightness,
        tint[1] * brightness,
        tint[2] * brightness,
    ]
}

/// Expands a mesh into two triangles per quad, in chunk-local space.
///
/// The triangles are wound counter-clockwise when seen from outside the face,
/// which is what the pipeline's `FrontFace::Ccw` with back-face culling needs;
/// getting this backwards makes every solid appear inside-out.
pub fn build_vertices(mesh: &Mesh) -> Vec<Vertex> {
    let mut out = Vec::with_capacity(mesh.quads.len() * 6);
    for quad in &mesh.quads {
        append_quad(&mut out, quad);
    }
    out
}

fn append_quad(out: &mut Vec<Vertex>, quad: &Quad) {
    let corners = quad.corners();
    let light = quad.light.sunlight().max(quad.light.block_light()).raw();
    let color = shade(quad.block, quad.dir, light);

    let v = |p: [f32; 3]| Vertex {
        position: p,
        color,
        _pad: [0.0; 2],
    };

    // Corners are (0,0), (u,0), (u,v), (0,v) in plane coordinates. Walking them
    // in that order yields the normal `u_axis x v_axis`, and whether that
    // matches the face's outward normal depends on which axes the plane uses:
    // it alternates per direction and cannot be captured by one sign rule, so
    // the cases that need reversing are listed explicitly and the winding test
    // checks all six.
    let [c0, c1, c2, c3] = corners;
    let (a, b, d, e) = if needs_reversed_winding(quad.dir) {
        (c0, c3, c2, c1)
    } else {
        (c0, c1, c2, c3)
    };

    // Triangle 1: a, b, d. Triangle 2: a, d, e.
    out.push(v(a));
    out.push(v(b));
    out.push(v(d));
    out.push(v(a));
    out.push(v(d));
    out.push(v(e));
}

/// True when walking a quad's corners in plane order yields an inward normal.
///
/// Derived from `u_axis x v_axis` against each direction's outward normal; the
/// three entries that need reversing are `NegX`, `PosY` and `NegZ`.
#[inline]
pub const fn needs_reversed_winding(dir: Direction) -> bool {
    matches!(dir, Direction::NegX | Direction::PosY | Direction::NegZ)
}

/// Vertex data for one chunk, with its world-space origin.
#[derive(Debug, Clone)]
pub struct ChunkMeshData {
    pub pos: ChunkPos,
    pub origin: [f32; 3],
    pub vertices: Vec<Vertex>,
}

impl ChunkMeshData {
    pub fn new(pos: ChunkPos, mesh: &Mesh) -> Self {
        let (ox, oy, oz) = pos.origin();
        Self {
            pos,
            origin: [ox as f32, oy as f32, oz as f32],
            vertices: build_vertices(mesh),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::mesh::mesh_chunk;
    use voxel_core::world::Chunk;

    fn solid() -> Chunk {
        let mut c = Chunk::at(ChunkPos::ORIGIN);
        c.fill(BlockId::STONE);
        c
    }

    #[test]
    fn vertex_is_pod_and_correctly_strided() {
        assert_eq!(std::mem::size_of::<Vertex>(), 32);
        assert_eq!(Vertex::LAYOUT.array_stride, 32);
        // The two attributes must not overlap and must fit the stride.
        let attrs = Vertex::LAYOUT.attributes;
        assert_eq!(attrs.len(), 2);
        assert_eq!(attrs[0].offset, 0);
        assert_eq!(attrs[1].offset, 12);
        assert_eq!(attrs[0].shader_location, 0);
        assert_eq!(attrs[1].shader_location, 1);
    }

    #[test]
    fn each_quad_becomes_two_triangles() {
        let mesh = mesh_chunk(&solid());
        let verts = build_vertices(&mesh);
        assert_eq!(verts.len(), mesh.quads.len() * 6, "6 vertices per quad");
    }

    #[test]
    fn winding_is_outward_for_every_direction() {
        // A solid cube: for each face, the triangle normal must point away from
        // the chunk centre. A reversed winding makes back-face culling hide the
        // outside of the world, which is the classic inside-out bug.
        let mesh = mesh_chunk(&solid());
        let verts = build_vertices(&mesh);
        assert_eq!(verts.len(), 36);

        let centre = [8.0f32, 8.0, 8.0];
        for tri in verts.chunks(3) {
            let [a, b, c] = [tri[0].position, tri[1].position, tri[2].position];
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let mid = [
                (a[0] + b[0] + c[0]) / 3.0,
                (a[1] + b[1] + c[1]) / 3.0,
                (a[2] + b[2] + c[2]) / 3.0,
            ];
            let outward = [mid[0] - centre[0], mid[1] - centre[1], mid[2] - centre[2]];
            let dot = n[0] * outward[0] + n[1] * outward[1] + n[2] * outward[2];
            assert!(
                dot > 0.0,
                "triangle winding faces inward: normal {n:?} vs outward {outward:?}"
            );
        }
    }

    #[test]
    fn top_faces_are_brighter_than_bottom_faces() {
        assert!(face_shade(Direction::PosY) > face_shade(Direction::NegY));
        assert!(face_shade(Direction::PosY) > face_shade(Direction::PosX));
        assert!(face_shade(Direction::NegX) > face_shade(Direction::NegZ));
    }

    #[test]
    fn shading_responds_to_light_level() {
        let dark = shade(BlockId::STONE, Direction::PosY, 0);
        let mid = shade(BlockId::STONE, Direction::PosY, 8);
        let bright = shade(BlockId::STONE, Direction::PosY, 15);
        assert!(dark[0] < mid[0], "light 0 must be darker than 8");
        assert!(mid[0] < bright[0], "light 8 must be darker than 15");
        // Even unlit geometry keeps some shape.
        assert!(dark[0] > 0.0, "ambient must keep dark faces visible");
    }

    #[test]
    fn unknown_blocks_are_visibly_wrong() {
        assert_eq!(block_tint(BlockId::new(200)), [1.0, 0.0, 1.0]);
    }

    #[test]
    fn chunk_mesh_data_carries_its_world_origin() {
        let pos = ChunkPos::new(2, 3, -1);
        let mut c = Chunk::at(pos);
        c.fill(BlockId::GRASS);
        let data = ChunkMeshData::new(pos, &mesh_chunk(&c));
        assert_eq!(data.origin, [32.0, 48.0, -16.0]);
        assert_eq!(data.pos, pos);
        assert!(!data.is_empty());
    }

    #[test]
    fn an_empty_chunk_produces_no_vertices() {
        let data = ChunkMeshData::new(ChunkPos::ORIGIN, &mesh_chunk(&Chunk::at(ChunkPos::ORIGIN)));
        assert!(data.is_empty());
        assert_eq!(data.vertex_count(), 0);
    }

    #[test]
    fn vertices_stay_within_the_chunk_bounds() {
        // Local space means every vertex is inside 0..=16, which is what lets
        // the same mesh be reused for any chunk position.
        let mesh = mesh_chunk(&solid());
        for v in build_vertices(&mesh) {
            for (i, &c) in v.position.iter().enumerate() {
                assert!((0.0..=16.0).contains(&c), "axis {i} escaped the chunk: {c}");
            }
        }
    }
}
