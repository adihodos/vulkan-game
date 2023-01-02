use crate::{
    app_config::AppConfig,
    color_palettes::StdColors,
    draw_context::DrawContext,
    math,
    vk_renderer::{
        Cpu2GpuBuffer, GraphicsPipelineBuilder, GraphicsPipelineLayoutBuilder,
        ShaderModuleDescription, ShaderModuleSource, UniqueBuffer, UniqueGraphicsPipeline,
        UniqueImageWithView, UniqueSampler, VulkanRenderer,
    },
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
    ubo_transforms: Cpu2GpuBuffer<glm::Mat4>,
    vertex_buffer: Cpu2GpuBuffer<SpriteVertex>,
    index_buffer: Cpu2GpuBuffer<u16>,
    vertices_cpu: Vec<SpriteVertex>,
    indices_cpu: Vec<u16>,
    descriptor_sets: Vec<DescriptorSet>,
    pipeline: UniqueGraphicsPipeline,
    sampler: UniqueSampler,
    texture: UniqueImageWithView,
}

#[derive(Copy, Clone)]
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
    fn new(texture: &UniqueImageWithView, region: TextureRegion) -> Self {
        assert!(region.layer < texture.info().num_layers);
        //
        //
        let region = if region.width == 0 || region.height == 0 {
            TextureRegion {
                width: texture.info().width,
                height: texture.info().height,
                ..region
            }
        } else {
            region
        };

        let u = region.x as f32 / texture.info().width as f32;
        let v = region.y as f32 / texture.info().height as f32;
        let s = region.width as f32 / texture.info().width as f32;
        let t = region.height as f32 / texture.info().height as f32;

        TextureCoords {
            layer: region.layer,
            bottom_left: glm::vec2(u, v),
            top_left: glm::vec2(u, v + t),
            top_right: glm::vec2(u + s, v + t),
            bottom_right: glm::vec2(u, v + t),
        }
    }
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

    pub fn create(renderer: &VulkanRenderer, app_config: &AppConfig) -> Option<Self> {
        let ubo_transforms = Cpu2GpuBuffer::<glm::Mat4>::create(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
            1,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let vertex_buffer = Cpu2GpuBuffer::<SpriteVertex>::create(
            renderer,
            BufferUsageFlags::VERTEX_BUFFER,
            renderer.device_properties().limits.non_coherent_atom_size,
            Self::MAX_SPRITES as DeviceSize,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let index_buffer = Cpu2GpuBuffer::<u16>::create(
            renderer,
            BufferUsageFlags::INDEX_BUFFER,
            renderer.device_properties().limits.non_coherent_atom_size,
            (Self::MAX_SPRITES * 6) as DeviceSize,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let tex_load_work_pkg = renderer.create_work_package()?;
        let texture = UniqueImageWithView::from_ktx(
            renderer,
            &tex_load_work_pkg,
            app_config.engine.texture_path("ui/reticles/reticle.ktx2"),
        )?;

        renderer.push_work_package(tex_load_work_pkg);

        let sampler = UniqueSampler::new(
            renderer.graphics_device(),
            &ash::vk::SamplerCreateInfo::builder()
                .min_lod(0f32)
                .max_lod(1f32)
                .min_filter(Filter::LINEAR)
                .mag_filter(Filter::LINEAR)
                .mipmap_mode(SamplerMipmapMode::LINEAR)
                .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                .border_color(ash::vk::BorderColor::INT_OPAQUE_BLACK)
                .max_anisotropy(1f32)
                .build(),
        )?;

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
                        &app_config.engine.shader_path("sprites.vert.spv"),
                    ),
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(
                        &app_config.engine.shader_path("sprites.frag.spv"),
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
            .build(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                GraphicsPipelineLayoutBuilder::new()
                    .set(
                        0,
                        &[DescriptorSetLayoutBinding::builder()
                            .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .stage_flags(ShaderStageFlags::VERTEX)
                            .descriptor_count(1)
                            .binding(0)
                            .build()],
                    )
                    .set(
                        1,
                        &[DescriptorSetLayoutBinding::builder()
                            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .stage_flags(ShaderStageFlags::FRAGMENT)
                            .descriptor_count(1)
                            .binding(0)
                            .build()],
                    )
                    .build(renderer.graphics_device())?,
                renderer.renderpass(),
                0,
            )?;

        let layouts = [
            pipeline.descriptor_layouts()[0],
            pipeline.descriptor_layouts()[1],
        ];

        let descriptor_sets = unsafe {
            renderer.graphics_device().allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .set_layouts(&layouts)
                    .descriptor_pool(renderer.descriptor_pool())
                    .build(),
            )
        }
        .map_err(|e| log::error!("Failed to allocate descriptor sets: {}", e))
        .ok()?;

        unsafe {
            renderer.graphics_device().update_descriptor_sets(
                &[
                    WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[0])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .buffer_info(&[DescriptorBufferInfo::builder()
                            .buffer(ubo_transforms.buffer.buffer)
                            .range(ubo_transforms.bytes_one_frame)
                            .offset(0)
                            .build()])
                        .build(),
                    WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[1])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&[DescriptorImageInfo::builder()
                            .sampler(sampler.sampler)
                            .image_view(texture.image_view())
                            .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .build()])
                        .build(),
                ],
                &[],
            );
        }

        Some(Self {
            ubo_transforms,
            vertex_buffer,
            index_buffer,
            vertices_cpu: Vec::with_capacity(Self::MAX_SPRITES as usize),
            indices_cpu: Vec::with_capacity((Self::MAX_SPRITES * 6) as usize),
            descriptor_sets,
            pipeline,
            sampler,
            texture,
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
        let texcoords = TextureCoords::new(&self.texture, region);
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

        let texcoords = TextureCoords::new(&self.texture, region);
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
        let (cost, sint) = rotation.sin_cos();
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

        let texcoords = TextureCoords::new(&self.texture, region);
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

        self.vertex_buffer
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|vb| unsafe {
                std::ptr::copy_nonoverlapping(
                    self.vertices_cpu.as_ptr(),
                    vb.memptr() as *mut SpriteVertex,
                    self.vertices_cpu.len(),
                );
            });

        self.index_buffer
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|ib| unsafe {
                std::ptr::copy_nonoverlapping(
                    self.indices_cpu.as_ptr(),
                    ib.memptr() as *mut u16,
                    self.indices_cpu.len(),
                );
            });

        self.ubo_transforms
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|ubo| {
                let ortho = math::orthographic(
                    0f32,
                    draw_context.viewport.width,
                    0f32,
                    draw_context.viewport.height,
                    1f32,
                    0f32,
                );

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &ortho as *const _,
                        ubo.memptr() as *mut glm::Mat4,
                        1,
                    );
                }
            });

        let graphics_device = draw_context.renderer.graphics_device();
        let cmd_buf = draw_context.cmd_buff;

        unsafe {
            graphics_device.cmd_bind_pipeline(
                cmd_buf,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            graphics_device.cmd_set_viewport(cmd_buf, 0, &[draw_context.viewport]);
            graphics_device.cmd_set_scissor(cmd_buf, 0, &[draw_context.scissor]);
            graphics_device.cmd_bind_vertex_buffers(
                cmd_buf,
                0,
                &[self.vertex_buffer.buffer.buffer],
                &[self
                    .vertex_buffer
                    .offset_for_frame(draw_context.frame_id as DeviceSize)],
            );
            graphics_device.cmd_bind_index_buffer(
                cmd_buf,
                self.index_buffer.buffer.buffer,
                self.index_buffer
                    .offset_for_frame(draw_context.frame_id as DeviceSize),
                IndexType::UINT16,
            );

            graphics_device.cmd_bind_descriptor_sets(
                cmd_buf,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &self.descriptor_sets,
                &[self
                    .ubo_transforms
                    .offset_for_frame(draw_context.frame_id as DeviceSize)
                    as u32],
            );

            graphics_device.cmd_draw_indexed(cmd_buf, self.indices_cpu.len() as u32, 1, 0, 0, 0);
        }

        self.vertices_cpu.clear();
        self.indices_cpu.clear();
    }
}
