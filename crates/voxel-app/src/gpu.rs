//! The wgpu renderer.
//!
//! One pipeline, one shader, one vertex buffer per chunk. Chunks are uploaded
//! once when their mesh is built; this POC has no dynamic geometry, so there is
//! no buffer recycling or streaming yet.
//!
//! Depth testing is on and back-face culling is enabled, which together are what
//! make the winding order in [`crate::render`] observable: get the winding wrong
//! and the world renders inside out, which the winding test prevents.

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

use crate::render::{ChunkMeshData, Vertex};

/// Depth format. `Depth32Float` is universally supported and has no precision
/// surprises at the far plane of a 2048-voxel view distance.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Colour format for offscreen rendering. sRGB so the captured image matches
/// what a windowed frame looks like.
pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Uniform block: the combined view-projection matrix.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_projection: [[f32; 4]; 4],
}

/// The shader, inline so the build has no asset path to get wrong.
///
/// Positions arrive in chunk-local space and the chunk origin is added from a
/// per-chunk uniform. wgpu's portable subset has no push constants, so a tiny
/// uniform buffer per chunk is the portable way to pass the origin.
const SHADER: &str = r#"
struct Camera {
    view_projection: mat4x4<f32>,
};

struct ChunkOffset {
    origin: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> chunk: ChunkOffset;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = vec4<f32>(in.position + chunk.origin.xyz, 1.0);
    out.clip_position = camera.view_projection * world;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

/// A chunk's GPU buffers plus how many vertices to draw.
struct GpuChunk {
    vertex_buffer: wgpu::Buffer,
    origin_bind_group: wgpu::BindGroup,
    vertex_count: u32,
}

/// Where a frame is drawn.
///
/// Both variants share the pipeline, buffers and drawing code, so an offscreen
/// frame exercises exactly the same path a windowed one does. That matters
/// because a headless machine cannot open a window at all, and an unverified
/// render path is worth very little.
enum Target {
    Surface(wgpu::Surface<'static>),
    /// Colour texture plus a staging buffer for readback.
    Offscreen {
        color: wgpu::Texture,
        readback: wgpu::Buffer,
        width: u32,
        height: u32,
    },
}

/// Owns the device, pipeline and all uploaded chunk geometry.
pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub adapter_info: wgpu::AdapterInfo,
    target: Target,
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    origin_layout: wgpu::BindGroupLayout,
    chunks: Vec<GpuChunk>,
    depth_view: wgpu::TextureView,
}

impl Renderer {
    /// Builds a renderer for an existing surface.
    /// The `instance` must be the one that created `surface`: wgpu tracks
    /// surfaces per instance, so creating the surface from one instance and the
    /// device from another fails at runtime with a "non-existent resource
    /// SurfaceId" panic rather than as a type error.
    pub async fn new(
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
        force_software: bool,
        vsync: bool,
    ) -> Result<Self> {

        // A software adapter is requested when no hardware one is reachable,
        // which is the usual case in a container or over a remote session.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: if force_software {
                    wgpu::PowerPreference::LowPower
                } else {
                    wgpu::PowerPreference::HighPerformance
                },
                compatible_surface: Some(&surface),
                force_fallback_adapter: force_software,
                ..Default::default()
            })
            .await
            .context(
                "no compatible GPU adapter; set VOXEL_FORCE_SOFTWARE_GPU=1 to use the CPU backend",
            )?;

        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("voxel-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                ..Default::default()
            })
            .await
            .context("requesting a GPU device")?;

        let caps = surface.get_capabilities(&adapter);
        // Prefer an sRGB target so the linear colours from `render::shade`
        // reach the display without a manual gamma step, but fall back to
        // whatever the surface offers since not every backend has one.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            // Immediate so an uncapped frame rate is measurable.
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::AutoNoVsync
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            // Auto lets the presentation engine pick, which is right for every
            // platform this POC targets.
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);

        let built = build_pipeline(&device, format)?;
        let depth_view = create_depth_view(&device, &config);

        Ok(Self {
            device,
            queue,
            config,
            adapter_info,
            target: Target::Surface(surface),
            pipeline: built.pipeline,
            camera_buffer: built.camera_buffer,
            camera_bind_group: built.camera_bind_group,
            origin_layout: built.origin_layout,
            chunks: Vec::new(),
            depth_view,
        })
    }

    /// Builds a headless renderer that draws into an offscreen texture.
    ///
    /// Shares `build_pipeline` with [`Renderer::new`], so a frame captured here
    /// is produced by the same shader, vertex layout and draw calls the window
    /// path uses.
    pub async fn new_offscreen(width: u32, height: u32, force_software: bool) -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: if force_software {
                    wgpu::PowerPreference::LowPower
                } else {
                    wgpu::PowerPreference::HighPerformance
                },
                compatible_surface: None,
                force_fallback_adapter: force_software,
                ..Default::default()
            })
            .await
            .context("no GPU adapter for offscreen rendering")?;
        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("voxel-device-offscreen"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                ..Default::default()
            })
            .await
            .context("requesting an offscreen device")?;

        let format = FORMAT;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };

        // Row padding: wgpu requires buffer rows to be a multiple of 256 bytes
        // when copying a texture out, so the readback buffer is wider than the
        // image and the padding is stripped when writing the file.
        let padded_row = padded_bytes_per_row(width, 4);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen-color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let built = build_pipeline(&device, format)?;
        let depth_view = create_depth_view(&device, &config);

        Ok(Self {
            device,
            queue,
            config,
            adapter_info,
            target: Target::Offscreen {
                color,
                readback,
                width,
                height,
            },
            pipeline: built.pipeline,
            camera_buffer: built.camera_buffer,
            camera_bind_group: built.camera_bind_group,
            origin_layout: built.origin_layout,
            chunks: Vec::new(),
            depth_view,
        })
    }

    /// Records the render pass shared by the surface and offscreen paths.
    ///
    /// Both targets draw the same chunks with the same pipeline, so any bug in
    /// the draw sequence shows up in a captured image too.
    fn encode_draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        clear: wgpu::Color,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        for chunk in &self.chunks {
            pass.set_bind_group(1, &chunk.origin_bind_group, &[]);
            pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
            pass.draw(0..chunk.vertex_count, 0..1);
        }
    }

    /// Draws one frame offscreen and returns it as RGB bytes.
    ///
    /// Errors if this renderer was created for a surface, since there is no
    /// texture to read back in that case.
    pub fn render_to_rgb(&mut self, clear: wgpu::Color) -> Result<(u32, u32, Vec<u8>)> {
        let Target::Offscreen {
            color,
            readback,
            width,
            height,
        } = &self.target
        else {
            anyhow::bail!("render_to_rgb requires an offscreen renderer");
        };
        let (width, height) = (*width, *height);
        let padded_row = padded_bytes_per_row(width, 4);

        let view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("offscreen-encoder"),
            });

        self.encode_draw(&mut encoder, &view, clear);

        // Copy the rendered texture into the mappable buffer, then pull it back.
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        readback.slice(..).map_async(wgpu::MapMode::Read, |r| {
            if let Err(e) = r {
                log::error!("readback mapping failed: {e}");
            }
        });
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .context("waiting for the offscreen frame")?;

        let data = readback
            .slice(..)
            .get_mapped_range()
            .context("mapping the readback buffer")?;
        // Strip the row padding so the result is a tight RGB image.
        let mut rgb = Vec::with_capacity((width * height * 3) as usize);
        for row in 0..height {
            let start = (row * padded_row) as usize;
            for px in 0..width as usize {
                let i = start + px * 4;
                rgb.extend_from_slice(&data[i..i + 3]);
            }
        }
        drop(data);
        readback.unmap();

        Ok((width, height, rgb))
    }

    /// Uploads a batch of chunk meshes, replacing anything already uploaded.
    pub fn upload_chunks(&mut self, meshes: &[ChunkMeshData]) {
        self.chunks.clear();
        for data in meshes {
            if data.is_empty() {
                // An empty mesh is skipped rather than kept as a zero-length
                // buffer, which would only add draw calls that do nothing.
                continue;
            }
            let vertex_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("chunk-vertices"),
                    contents: bytemuck::cast_slice(&data.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

            // `vec4` in WGSL so the uniform meets the 16-byte alignment rule.
            let origin = [data.origin[0], data.origin[1], data.origin[2], 0.0];
            let origin_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("chunk-origin"),
                    contents: bytemuck::cast_slice(&origin),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let origin_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chunk-origin-bind-group"),
                layout: &self.origin_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: origin_buffer.as_entire_binding(),
                }],
            });

            self.chunks.push(GpuChunk {
                vertex_buffer,
                origin_bind_group,
                vertex_count: data.vertices.len() as u32,
            });
        }
    }

    /// Total vertices uploaded, for the debug overlay.
    pub fn uploaded_vertices(&self) -> usize {
        self.chunks.iter().map(|c| c.vertex_count as usize).sum()
    }

    /// Number of draw calls the next frame will issue.
    pub fn uploaded_chunks(&self) -> usize {
        self.chunks.len()
    }

    pub fn set_view_projection(&self, view_projection: [[f32; 4]; 4]) {
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform { view_projection }),
        );
    }

    /// Resizes the surface and its depth texture.
    pub fn resize(&mut self, width: u32, height: u32) {
        // A minimised window reports zero, and configuring a surface with a
        // zero extent is a validation error, so keep the previous size.
        if width == 0 || height == 0 {
            return;
        }
        if width == self.config.width && height == self.config.height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        if let Target::Surface(surface) = &self.target {
            surface.configure(&self.device, &self.config);
        }
        self.depth_view = create_depth_view(&self.device, &self.config);
    }

    /// Clears to sky and draws every uploaded chunk.
    ///
    /// Returns `Ok(())` for a frame that was drawn, `Err(FrameOutcome::Skip)`
    /// for a frame the compositor did not need, and `Err(FrameOutcome::Reconfigure)`
    /// when the surface must be rebuilt before the next attempt.
    pub fn render(&mut self, clear: wgpu::Color) -> Result<(), FrameOutcome> {
        // wgpu 30 returns a status enum rather than a Result, and presenting
        // moved from the texture onto the queue.
        // An offscreen renderer has no swap chain, so there is nothing to
        // present; the caller should be using `render_to_rgb`.
        let Target::Surface(surface) = &self.target else {
            return Err(FrameOutcome::Skip);
        };
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                // Nothing to draw: the window is hidden or the compositor is
                // not ready. Skipping the frame is correct, not an error.
                return Err(FrameOutcome::Skip);
            }
            wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Validation => return Err(FrameOutcome::Reconfigure),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-encoder"),
            });
        self.encode_draw(&mut encoder, &view, clear);

        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);
        Ok(())
    }
}

/// What a failed frame means for the caller's next step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameOutcome {
    /// No frame was produced and the loop should simply continue.
    Skip,
    /// The surface must be reconfigured before drawing again.
    Reconfigure,
}

impl std::fmt::Display for FrameOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Skip => "frame skipped",
            Self::Reconfigure => "surface needs reconfiguring",
        })
    }
}

impl std::error::Error for FrameOutcome {}


/// The parts of a renderer that do not depend on where frames are drawn.
struct PipelineParts {
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    origin_layout: wgpu::BindGroupLayout,
}

/// Builds the shader, layouts and pipeline.
///
/// Shared by the surface and offscreen constructors so both paths cannot drift
/// apart; a bug fixed in one is fixed in the other by construction.
fn build_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> Result<PipelineParts> {
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera-uniform"),
            contents: bytemuck::bytes_of(&CameraUniform {
                view_projection: IDENTITY,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_layout = uniform_layout(device, "camera-layout");
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-bind-group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let origin_layout = uniform_layout(device, "chunk-origin-layout");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-pipeline-layout"),
            bind_group_layouts: &[Some(&camera_layout), Some(&origin_layout)],
            // wgpu 30 replaced push constants with an immediate-data size.
            // Nothing here uses either, so zero.
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("voxel-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(Vertex::LAYOUT)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Back-face culling is what makes the winding order matter.
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

    Ok(PipelineParts {
        pipeline,
        camera_buffer,
        camera_bind_group,
        origin_layout,
    })
}

/// Rounds a row length up to wgpu's 256-byte alignment requirement for texture
/// reads, which is why captured images need their padding stripped.
fn padded_bytes_per_row(width: u32, bytes_per_pixel: u32) -> u32 {
    let unpadded = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    unpadded.div_ceil(align) * align
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// A single-uniform bind group layout, used for both the camera and a chunk
/// origin; they differ only in binding name.
fn uniform_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn create_depth_view(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth-texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::mesh::mesh_chunk;
    use voxel_core::world::Chunk;
    use voxel_core::{BlockId, ChunkPos};

    #[test]
    fn camera_uniform_is_sixty_four_bytes() {
        // WGSL requires a mat4x4 uniform to occupy exactly 64 bytes; a mismatch
        // is a runtime validation error rather than a compile error.
        assert_eq!(std::mem::size_of::<CameraUniform>(), 64);
    }

    #[test]
    fn shader_parses_as_wgsl() {
        // A syntax error in the inline shader would otherwise only surface when
        // a real device is created, which the unit tests deliberately avoid.
        let module = naga::front::wgsl::parse_str(SHADER);
        assert!(module.is_ok(), "shader failed to parse: {:?}", module.err());
    }

    #[test]
    fn shader_declares_both_entry_points() {
        let module = naga::front::wgsl::parse_str(SHADER).expect("shader parses");
        let names: Vec<&str> = module.entry_points.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"vs_main"), "missing vertex entry: {names:?}");
        assert!(names.contains(&"fs_main"), "missing fragment entry: {names:?}");
    }

    #[test]
    fn shader_vertex_inputs_cover_the_vertex_layout() {
        // The buffer layout and the shader live in different places; if their
        // locations drift apart the pipeline fails when it is created.
        let module = naga::front::wgsl::parse_str(SHADER).expect("shader parses");
        let vs = module
            .entry_points
            .iter()
            .find(|e| e.name == "vs_main")
            .expect("vs_main");
        // `vs_main` takes a struct, so naga attaches the locations to the
        // struct's members rather than to the argument itself.
        let declared: Vec<u32> = vs
            .function
            .arguments
            .iter()
            .flat_map(|arg| match &module.types[arg.ty].inner {
                naga::TypeInner::Struct { members, .. } => members
                    .iter()
                    .filter_map(|m| match m.binding.as_ref() {
                        Some(naga::Binding::Location { location, .. }) => Some(*location),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect();
        for attr in Vertex::LAYOUT.attributes {
            assert!(
                declared.contains(&attr.shader_location),
                "vertex layout uses location {} but the shader does not declare it",
                attr.shader_location
            );
        }
    }

    #[test]
    fn vertex_data_casts_to_bytes_with_the_expected_stride() {
        // Exercises the bytemuck path the uploader depends on, with no device.
        let mut c = Chunk::at(ChunkPos::ORIGIN);
        c.fill(BlockId::STONE);
        let data = ChunkMeshData::new(ChunkPos::ORIGIN, &mesh_chunk(&c));
        let bytes: &[u8] = bytemuck::cast_slice(&data.vertices);
        assert_eq!(bytes.len(), data.vertices.len() * 32);
    }
}
