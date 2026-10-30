# High-Performance Voxel Engine Benchmark Architecture

## 1. Project Goal

Build a ultra-performant, cross-platform voxel engine prototype in Rust to benchmark two cave generation paradigms:

1. **Full 3D Density Field ($O(N^3)$)**: Standard continuous 3D noise evaluation per voxel.
2. **Sparse Cave Stamping ($O(S)$)**: Deterministic origin-point noise combined with localized SDF (Signed Distance Field) stamp carving.

## 2. Core Architecture & Design Philosophy

### Data-Oriented Design (DOD)

- **Structure of Arrays (SoA)** over Array of Structures (AoS).
- Linear, cache-line-friendly contiguity ($64$-byte aligned buffers).
- Zero virtual dispatch or heavy polymorphic allocations in execution hot-paths.
- Minimal pointer indirection; prefer flat index lookup tables.

### Parse, Don't Validate

- **Type-Level Correctness**: Eliminate runtime validation checks inside inner loops.
- Bit-packed domain types (e.g., `ChunkCoord`, `VoxelIndex`, `PackedVoxel`) ensure impossible states cannot be represented at compile time.
- Data transformations convert raw input into verified structs once during instantiation.

### Bit-Packing & Memory Optimization

- Chunk dimensions: $16 \times 16 \times 16$ ($A = 16 = 2^4$).
- Voxel indices derived exclusively via bit-shifts (`x | (z << 4) | (y << 8)`).
- Bit-packed voxel definitions (`u16` or `u32` storing ID, lighting flags, and occlusion states).

## 3. Technology Stack & Minimal Dependencies

| Category             | Library / Tech                          | Purpose                                                          |
| :------------------- | :-------------------------------------- | :--------------------------------------------------------------- |
| **Language**         | Rust (Nightly for `std::simd` / Stable) | Memory safety, zero-cost abstractions, low-level control.        |
| **Graphics API**     | `wgpu`                                  | Modern cross-platform GPU abstraction (Vulkan, Metal, DX12).     |
| **Windowing**        | `winit`                                 | Cross-platform window context management.                        |
| **Concurrency**      | `tokio`                                 | Async task orchestration and background execution.               |
| **Data Parallelism** | `rayon`                                 | Multithreaded chunk generation and mesh building pipelines.      |
| **SIMD**             | `std::simd`                             | Vectorized math operations without external dependencies.        |
| **Benchmarking**     | `criterion`                             | Statistical microbenchmarking for $O(N^3)$ vs $O(S)$ comparison. |

## 4. Key Systems to Implement

1. **Chunk Storage**:
    - Sparse 3D hash map or spatial grid of $16^3$ cubic chunks.
    - Flat array buffers (`[PackedVoxel; 4096]`).
2. **Terrain Pipeline**:
    - **Pass 1**: 2D Surface Heightmap evaluation.
    - **Pass 2A (Baseline)**: $O(N^3)$ full 3D Simplex/Perlin noise density carver.
    - **Pass 2B (Target)**: $O(S)$ deterministic origin lookup + bounded SDF stamp carving with neighborhood padding.
3. **Frustum & Occlusion Culling**:
    - Complete skip for air-only chunks.
    - Cull subterranean fully enclosed solid chunks from rendering pipelines.
4. **Greedy Meshing**:
    - Convert face boundaries into optimized quad meshes via bitwise scanning.
