use crate::{
    app_config::AppConfig,
    bindless::BindlessResourceHandle2,
    color_palettes::StdColors,
    draw_context::{DrawContext, InitContext},
    math,
    vk_renderer::{
        BindlessPipeline, Cpu2GpuBuffer, GraphicsPipelineBuilder, GraphicsPipelineLayoutBuilder,
        ImageInfo, ShaderModuleDescription, ShaderModuleSource, UniqueBuffer,
        UniqueGraphicsPipeline, UniqueImageWithView, UniqueSampler, VulkanRenderer,
    },
    ProgramError,
};
use ash::vk::{
    BlendFactor, BlendOp, BufferUsageFlags, ColorComponentFlags, DescriptorBufferInfo,
    DescriptorImageInfo, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayoutBinding,
    DescriptorType, DeviceSize, DynamicState, Filter, Format, ImageLayout, IndexType,
    MemoryPropertyFlags, PipelineBindPoint, PipelineColorBlendAttachmentState, PrimitiveTopology,
    SamplerAddressMode, SamplerMipmapMode, ShaderStageFlags, VertexInputAttributeDescription,
    VertexInputBindingDescription, VertexInputRate, WriteDescriptorSet,
};
use memoffset::offset_of;
use nalgebra_glm as glm;

pub struct SpriteBatch {
    // ubo_transforms: Cpu2GpuBuffer<glm::Mat4>,
    vertex_buffer: UniqueBuffer,
    index_buffer: UniqueBuffer,
    vertices_cpu: Vec<SpriteVertex>,
    indices_cpu: Vec<u16>,
    // descriptor_sets: Vec<DescriptorSet>,
    pipeline: BindlessPipeline,
    // sampler: UniqueSampler,
    // texture: UniqueImageWithView,
    atlas: TextureAtlas,
    texture_atlas_handle: BindlessResourceHandle2,
    texture_info: ImageInfo,
}

#[derive(Copy, Clone, serde::Serialize, serde::Deserialize)]
struct NamedTextureRegion {
    pub name: u64,
    pub layer: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Copy, Clone, serde::Serialize, serde::Deserialize)]
pub struct TextureRegion {
    pub layer: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl TextureRegion {
    pub fn complete(layer: u32) -> TextureRegion {
        TextureRegion {
            layer,
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        }
    }
}

struct TextureCoords {
    layer: u32,
    bottom_left: glm::Vec2,
    top_left: glm::Vec2,
    top_right: glm::Vec2,
    bottom_right: glm::Vec2,
}

impl TextureCoords {
    fn new(texture: &ImageInfo, region: TextureRegion) -> Self {
        assert!(region.layer < texture.num_layers);
        //
        //
        let region = if region.width == 0 || region.height == 0 {
            TextureRegion {
                width: texture.width,
                height: texture.height,
                ..region
            }
        } else {
            region
        };

        let u = region.x as f32 / texture.width as f32;
        let v = region.y as f32 / texture.height as f32;
        let s = region.width as f32 / texture.width as f32;
        let t = region.height as f32 / texture.height as f32;

        TextureCoords {
            layer: region.layer,
            bottom_left: glm::vec2(u, v),
            top_left: glm::vec2(u, v + t),
            top_right: glm::vec2(u + s, v + t),
            bottom_right: glm::vec2(u, v + t),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TextureAtlas {
    frames: Vec<NamedTextureRegion>,
    size: (u32, u32),
    file: std::path::PathBuf,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct SpriteVertex {
    pos: glm::Vec2,
    uv: glm::Vec2,
    texid: u32,
    color: u32,
}

impl SpriteBatch {
    const MAX_SPRITES: u32 = 2048;

    pub fn new(init_ctx: &mut InitContext) -> Result<Self, ProgramError> {
        let vertex_buffer = UniqueBuffer::with_capacity(
            init_ctx.renderer,
            BufferUsageFlags::VERTEX_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            Self::MAX_SPRITES as usize,
            std::mem::size_of::<SpriteVertex>(),
            init_ctx.renderer.max_inflight_frames(),
        )?;

        let index_buffer = UniqueBuffer::with_capacity(
            init_ctx.renderer,
            BufferUsageFlags::INDEX_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            (Self::MAX_SPRITES * 6) as usize,
            std::mem::size_of::<u16>(),
            init_ctx.renderer.max_inflight_frames(),
        )?;

        let texture_atlas: TextureAtlas = ron::de::from_reader(
            std::fs::File::open(
                init_ctx
                    .cfg
                    .engine
                    .texture_path("ui/reticles/crosshairs.ron"),
            )
            .expect("Failed to read texture atlas configuration file."),
        )
        .expect("Invalid configuration file");

        let tex_load_work_pkg = init_ctx
            .renderer
            .create_work_package()
            .ok_or_else(|| ProgramError::GraphicsSystemError(ash::vk::Result::ERROR_UNKNOWN))?;

        let atlas_texture = UniqueImageWithView::from_ktx(
            init_ctx.renderer,
            &tex_load_work_pkg,
            init_ctx
                .cfg
                .engine
                .texture_path(std::path::Path::new("ui/reticles").join(&texture_atlas.file)),
        )
        .ok_or_else(|| ProgramError::GraphicsSystemError(ash::vk::Result::ERROR_UNKNOWN))?;
        init_ctx.renderer.push_work_package(tex_load_work_pkg);

        let texture_info = *atlas_texture.info();

        let texture_atlas_handle = init_ctx.rsys.add_texture_bindless(
            "ui/reticles",
            init_ctx.renderer,
            atlas_texture,
            None,
        );

        let pipeline = GraphicsPipelineBuilder::new()
            .add_vertex_input_attribute_descriptions(&[
                VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: Format::R32G32_SFLOAT,
                    offset: offset_of!(SpriteVertex, pos) as u32,
                },
                VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: Format::R32G32_SFLOAT,
                    offset: offset_of!(SpriteVertex, uv) as u32,
                },
                VertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: Format::R32_UINT,
                    offset: offset_of!(SpriteVertex, texid) as u32,
                },
                VertexInputAttributeDescription {
                    location: 3,
                    binding: 0,
                    format: Format::R8G8B8A8_UNORM,
                    offset: offset_of!(SpriteVertex, color) as u32,
                },
            ])
            .add_vertex_input_attribute_binding(
                VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(std::mem::size_of::<SpriteVertex>() as u32)
                    .input_rate(VertexInputRate::VERTEX)
                    .build(),
            )
            .set_input_assembly_state(PrimitiveTopology::TRIANGLE_LIST, false)
            .shader_stages(&[
                ShaderModuleDescription {
                    stage: ShaderStageFlags::VERTEX,
                    source: ShaderModuleSource::File(
                        &init_ctx.cfg.engine.shader_path("sprites.bindless.vert.spv"),
                    ),
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(
                        &init_ctx.cfg.engine.shader_path("sprites.bindless.frag.spv"),
                    ),
                    entry_point: "main",
                },
            ])
            .set_depth_test(false)
            .set_colorblend_attachment(
                0,
                PipelineColorBlendAttachmentState::builder()
                    .blend_enable(true)
                    .color_blend_op(BlendOp::ADD)
                    .alpha_blend_op(BlendOp::ADD)
                    .src_color_blend_factor(BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .src_alpha_blend_factor(BlendFactor::ONE)
                    .dst_alpha_blend_factor(BlendFactor::ZERO)
                    .color_write_mask(ColorComponentFlags::RGBA)
                    .build(),
            )
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR])
            .build_bindless(
                init_ctx.renderer.graphics_device(),
                init_ctx.renderer.pipeline_cache(),
                init_ctx.rsys.bindless_setup().pipeline_layout,
                init_ctx.renderer.renderpass(),
                0,
            )?;

        Ok(Self {
            texture_info,
            vertex_buffer,
            index_buffer,
            texture_atlas_handle,
            atlas: texture_atlas,
            pipeline,
            vertices_cpu: vec![],
            indices_cpu: vec![],
        })
    }

    pub fn draw(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        region: TextureRegion,
        color: Option<u32>,
    ) {
        let texcoords = TextureCoords::new(&self.texture_info, region);
        let sprite_color = color.unwrap_or(StdColors::WHITE);
        let vertex_offset = self.vertices_cpu.len() as u16;

        self.vertices_cpu.extend_from_slice(&[
            SpriteVertex {
                pos: glm::vec2(x, y + height),
                uv: texcoords.bottom_left,
                color: sprite_color,
                texid: region.layer,
            },
            SpriteVertex {
                pos: glm::vec2(x, y),
                uv: texcoords.top_left,
                color: sprite_color,
                texid: region.layer,
            },
            SpriteVertex {
                pos: glm::vec2(x + width, y),
                uv: texcoords.top_right,
                color: sprite_color,
                texid: region.layer,
            },
            SpriteVertex {
                pos: glm::vec2(x + width, y + height),
                uv: texcoords.bottom_right,
                color: sprite_color,
                texid: region.layer,
            },
        ]);

        self.indices_cpu.extend(
            [0, 2, 1, 0, 3, 2]
                .into_iter()
                .map(|idx| idx as u16 + vertex_offset),
        );
    }

    pub fn draw_scaled_rotated(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        scale: f32,
        rotation: f32,
        region: TextureRegion,
        color: Option<u32>,
    ) {
        let v0 = glm::vec2(x, y + height);
        let v1 = glm::vec2(x, y);
        let v2 = glm::vec2(x + width, y);
        let v3 = glm::vec2(x + width, y + height);

        //
        // translate to origin
        let o = glm::vec2(x + width * 0.5f32, y + height * 0.5f32);
        let v0 = v0 - o;
        let v1 = v1 - o;
        let v2 = v2 - o;
        let v3 = v3 - o;

        //
        // scale
        let s = scale;
        let v0 = s * v0;
        let v1 = s * v1;
        let v2 = s * v2;
        let v3 = s * v3;

        //
        // rotate
        let (sin_theta, cos_theta) = rotation.sin_cos();
        let v0 = glm::vec2(
            v0.x * cos_theta - v0.y * sin_theta,
            v0.x * sin_theta + v0.y * cos_theta,
        );
        let v1 = glm::vec2(
            v1.x * cos_theta - v1.y * sin_theta,
            v1.x * sin_theta + v1.y * cos_theta,
        );
        let v2 = glm::vec2(
            v2.x * cos_theta - v2.y * sin_theta,
            v2.x * sin_theta + v2.y * cos_theta,
        );
        let v3 = glm::vec2(
            v3.x * cos_theta - v3.y * sin_theta,
            v3.x * sin_theta + v3.y * cos_theta,
        );

        //
        // translate back
        let v0 = v0 + o;
        let v1 = v1 + o;
        let v2 = v2 + o;
        let v3 = v3 + o;

        let texcoords = TextureCoords::new(&self.texture_info, region);
        let sprite_color = color.unwrap_or(StdColors::WHITE);
        let vertex_offset = self.vertices_cpu.len() as u16;

        self.vertices_cpu.extend_from_slice(&[
            SpriteVertex {
                pos: v0,
                uv: texcoords.bottom_left,
                color: sprite_color,
                texid: texcoords.layer,
            },
            SpriteVertex {
                pos: v1,
                uv: texcoords.top_left,
                color: sprite_color,
                texid: texcoords.layer,
            },
            SpriteVertex {
                pos: v2,
                uv: texcoords.top_right,
                color: sprite_color,
                texid: texcoords.layer,
            },
            SpriteVertex {
                pos: v3,
                uv: texcoords.bottom_right,
                color: sprite_color,
                texid: texcoords.layer,
            },
        ]);

        self.indices_cpu.extend(
            [0, 2, 1, 0, 3, 2]
                .into_iter()
                .map(|idx| idx as u16 + vertex_offset),
        );
    }

    pub fn draw_with_origin(
        &mut self,
        ox: f32,
        oy: f32,
        width: f32,
        height: f32,
        region: TextureRegion,
        color: Option<u32>,
    ) {
        let (hw, hh) = (width * 0.5f32, height * 0.5f32);
        self.draw(ox - hw, oy - hh, width, height, region, color);
    }

    pub fn draw_scaled_rotated_with_origin(
        &mut self,
        ox: f32,
        oy: f32,
        width: f32,
        height: f32,
        scale: f32,
        rotation: f32,
        region: TextureRegion,
        color: Option<u32>,
    ) {
        let t = glm::vec2(ox, oy);
        let (hw, hh) = (width * 0.5f32, height * 0.5f32);

        //
        // generate vertices from origin @ (0, 0)
        let v0 = glm::vec2(-hw, hh);
        let v1 = glm::vec2(-hw, -hh);
        let v2 = glm::vec2(hw, -hh);
        let v3 = glm::vec2(hw, hh);

        //
        // scale
        let v0 = v0 * scale;
        let v1 = v1 * scale;
        let v2 = v2 * scale;
        let v3 = v3 * scale;

        //
        // rotate
        let (sint, cost) = rotation.sin_cos();
        let v0 = glm::vec2(v0.x * cost - v0.y * sint, v0.x * sint + v0.y * cost);
        let v1 = glm::vec2(v1.x * cost - v1.y * sint, v1.x * sint + v1.y * cost);
        let v2 = glm::vec2(v2.x * cost - v2.y * sint, v2.x * sint + v2.y * cost);
        let v3 = glm::vec2(v3.x * cost - v3.y * sint, v3.x * sint + v3.y * cost);

        //
        // translate back to origin
        let v0 = v0 + t;
        let v1 = v1 + t;
        let v2 = v2 + t;
        let v3 = v3 + t;

        let texcoords = TextureCoords::new(&self.texture_info, region);
        let sprite_color = color.unwrap_or(StdColors::WHITE);
        let vertex_offset = self.vertices_cpu.len() as u16;
        self.vertices_cpu.extend_from_slice(&[
            SpriteVertex {
                pos: v0,
                uv: texcoords.bottom_left,
                color: sprite_color,
                texid: texcoords.layer,
            },
            SpriteVertex {
                pos: v1,
                uv: texcoords.top_left,
                color: sprite_color,
                texid: texcoords.layer,
            },
            SpriteVertex {
                pos: v2,
                uv: texcoords.top_right,
                color: sprite_color,
                texid: texcoords.layer,
            },
            SpriteVertex {
                pos: v3,
                uv: texcoords.bottom_right,
                color: sprite_color,
                texid: texcoords.layer,
            },
        ]);

        self.indices_cpu.extend(
            [0, 2, 1, 0, 3, 2]
                .into_iter()
                .map(|idx| idx as u16 + vertex_offset),
        );
    }

    pub fn render(&mut self, draw_context: &DrawContext) {
        if self.indices_cpu.is_empty() {
            return;
        }

        let _ = self
            .vertex_buffer
            .map_for_frame(draw_context.renderer, draw_context.frame_id)
            .map(|vb| unsafe {
                std::ptr::copy_nonoverlapping(
                    self.vertices_cpu.as_ptr(),
                    vb.memptr() as *mut SpriteVertex,
                    self.vertices_cpu.len(),
                );
            });

        let _ = self
            .index_buffer
            .map_for_frame(draw_context.renderer, draw_context.frame_id)
            .map(|ib| unsafe {
                std::ptr::copy_nonoverlapping(
                    self.indices_cpu.as_ptr(),
                    ib.memptr() as *mut u16,
                    self.indices_cpu.len(),
                );
            });

        let graphics_device = draw_context.renderer.graphics_device();
        let cmd_buf = draw_context.cmd_buff;

        unsafe {
            graphics_device.cmd_bind_pipeline(
                cmd_buf,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.handle,
            );
            graphics_device.cmd_set_viewport(cmd_buf, 0, &[draw_context.viewport]);
            graphics_device.cmd_set_scissor(cmd_buf, 0, &[draw_context.scissor]);
            graphics_device.cmd_bind_vertex_buffers(
                cmd_buf,
                0,
                &[self.vertex_buffer.buffer],
                &[
                    (self.vertex_buffer.aligned_slab_size * draw_context.frame_id as usize)
                        as DeviceSize,
                ],
            );
            graphics_device.cmd_bind_index_buffer(
                cmd_buf,
                self.index_buffer.buffer,
                (self.index_buffer.aligned_slab_size * draw_context.frame_id as usize)
                    as DeviceSize,
                IndexType::UINT16,
            );

            let push_const =
                (draw_context.global_ubo_handle << 16) | (self.texture_atlas_handle.handle());

            graphics_device.cmd_push_constants(
                draw_context.cmd_buff,
                self.pipeline.layout,
                ShaderStageFlags::ALL,
                0,
                &push_const.to_le_bytes(),
            );

            graphics_device.cmd_draw_indexed(cmd_buf, self.indices_cpu.len() as u32, 1, 0, 0, 0);
        }

        self.vertices_cpu.clear();
        self.indices_cpu.clear();
    }

    pub fn get_sprite_by_name(&self, name: &str) -> Option<TextureRegion> {
        use std::hash::Hasher;
        let mut h = fnv::FnvHasher::default();
        h.write(name.as_bytes());
        let hashed_name = h.finish();

        self.atlas
            .frames
            .iter()
            .find(|named_tex_region| named_tex_region.name == hashed_name)
            .map(|region| TextureRegion {
                layer: region.layer,
                x: region.x,
                y: region.y,
                width: region.width,
                height: region.height,
            })
    }
}
