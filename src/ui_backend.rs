use ash::vk::{
    BlendFactor, BlendOp, BorderColor, BufferUsageFlags, ColorComponentFlags, CommandBuffer,
    ComponentMapping, CopyDescriptorSet, CullModeFlags, DescriptorBufferInfo, DescriptorImageInfo,
    DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayoutBinding, DescriptorType, Device,
    DeviceSize, DynamicState, Extent2D, Extent3D, Filter, Format, FrontFace, Handle,
    ImageAspectFlags, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageTiling, ImageType,
    ImageUsageFlags, ImageViewCreateInfo, ImageViewType, IndexType, MemoryPropertyFlags,
    MemoryType, Offset2D, PipelineBindPoint, PipelineColorBlendAttachmentState,
    PipelineRasterizationStateCreateInfo, PolygonMode, PrimitiveTopology, PushConstantRange,
    Rect2D, SampleCountFlags, SampleMask, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
    ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, VertexInputAttributeDescription,
    VertexInputBindingDescription, VertexInputRate, WriteDescriptorSet,
};
use glfw::{Action, Cursor, Modifiers, MouseButton, WindowEvent};
use imgui::{draw_list, DrawCmd, DrawVert, TextureId};
use log::{error, info};
use memoffset::offset_of;
use nalgebra_glm::{I32Vec2, Vec2};

use crate::vk_renderer::{
    self, DrawContext, FrameRenderData, GraphicsPipelineBuilder, GraphicsPipelineLayoutBuilder,
    ImageCopySource, ScopedBufferMapping, ShaderModuleDescription, ShaderModuleSource,
    UniqueBuffer, UniqueGraphicsPipeline, UniqueImage, UniqueImageView, UniqueSampler,
    VulkanRenderer,
};

use std::{
    cell::{Cell, RefCell, RefMut},
    mem::size_of,
    path::Path,
    ptr::slice_from_raw_parts,
};

type UiVertex = imgui::DrawVert;
type UiIndex = imgui::DrawIdx;

#[repr(C)]
struct Uniform {
    world_view_proj: nalgebra_glm::Mat4,
}

struct ImguiGlfwData {
    time: f64,
    cursors: Vec<Cursor>,
    last_valid_mousepos: Vec2,
}

impl ImguiGlfwData {
    pub fn new(window: &glfw::Window) -> Self {
        let cursors = imgui::MouseCursor::VARIANTS
            .iter()
            .map(|imgui_cursor| {
                let glfw_cursor = match imgui_cursor {
                    imgui::MouseCursor::Arrow => glfw::StandardCursor::Arrow,
                    imgui::MouseCursor::TextInput => glfw::StandardCursor::IBeam,
                    imgui::MouseCursor::ResizeNS => glfw::StandardCursor::VResize,
                    imgui::MouseCursor::ResizeEW => glfw::StandardCursor::HResize,
                    imgui::MouseCursor::Hand => glfw::StandardCursor::Hand,
                    imgui::MouseCursor::ResizeAll => glfw::StandardCursor::Arrow,
                    imgui::MouseCursor::ResizeNESW => glfw::StandardCursor::Arrow,
                    imgui::MouseCursor::ResizeNWSE => glfw::StandardCursor::Arrow,
                    imgui::MouseCursor::NotAllowed => glfw::StandardCursor::Arrow,
                };

                glfw::Cursor::standard(glfw_cursor)
            })
            .collect::<Vec<_>>();

        ImguiGlfwData {
            time: 0f64,
            cursors,
            last_valid_mousepos: Vec2::new(f32::MAX, f32::MAX),
        }
    }

    fn glfw_key2imgui_hey(key: glfw::Key) -> imgui::Key {
        match key {
            glfw::Key::Tab => imgui::Key::Tab,
            glfw::Key::Left => imgui::Key::LeftArrow,
            glfw::Key::Right => imgui::Key::RightArrow,
            glfw::Key::Up => imgui::Key::UpArrow,
            glfw::Key::Down => imgui::Key::DownArrow,
            glfw::Key::PageUp => imgui::Key::PageUp,
            glfw::Key::PageDown => imgui::Key::PageDown,
            glfw::Key::Home => imgui::Key::Home,
            glfw::Key::End => imgui::Key::End,
            glfw::Key::Insert => imgui::Key::Insert,
            glfw::Key::Delete => imgui::Key::Delete,
            glfw::Key::Backspace => imgui::Key::Backspace,
            glfw::Key::Space => imgui::Key::Space,
            glfw::Key::Enter => imgui::Key::Enter,
            glfw::Key::Escape => imgui::Key::Escape,
            glfw::Key::A => imgui::Key::A,
            glfw::Key::C => imgui::Key::C,
            glfw::Key::V => imgui::Key::V,
            glfw::Key::X => imgui::Key::X,
            glfw::Key::Y => imgui::Key::Y,
            glfw::Key::Z => imgui::Key::Z,
            _ => imgui::Key::Escape,
        }
    }
}

pub struct UiBackend {
    imgui: RefCell<imgui::Context>,
    vertex_bytes_one_frame: DeviceSize,
    index_bytes_one_frame: DeviceSize,
    ubo_bytes_one_frame: DeviceSize,
    window_size: Cell<I32Vec2>,
    framebuffer_size: Cell<I32Vec2>,
    uniform_buffer: UniqueBuffer,
    sampler: UniqueSampler,
    pipeline: UniqueGraphicsPipeline,
    descriptor_set: DescriptorSet,
    vertex_buffer: UniqueBuffer,
    index_buffer: UniqueBuffer,
    font_atlas_image: UniqueImage,
    font_atlas_imageview: UniqueImageView,
}

impl UiBackend {
    const MAX_VERTICES: u32 = 8192;
    const MAX_INDICES: u32 = 16535;

    pub fn new(renderer: &VulkanRenderer, window: &glfw::Window) -> Option<UiBackend> {
        info!(
            "UI vertex type size = {}, index type size = {}",
            std::mem::size_of::<UiVertex>(),
            std::mem::size_of::<UiIndex>()
        );

        let vertex_bytes_one_frame = UiBackend::MAX_VERTICES
            * VulkanRenderer::aligned_size_of_type::<UiVertex>(
                renderer.device_properties().limits.non_coherent_atom_size,
            ) as u32;

        let vertex_buffer_size = vertex_bytes_one_frame * renderer.max_inflight_frames();
        let vertex_buffer = UniqueBuffer::new(
            renderer,
            BufferUsageFlags::VERTEX_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            vertex_buffer_size as DeviceSize,
        )?;

        let index_bytes_one_frame = UiBackend::MAX_INDICES
            * VulkanRenderer::aligned_size_of_type::<UiIndex>(
                renderer.device_properties().limits.non_coherent_atom_size,
            ) as u32;
        let index_buffer_size = index_bytes_one_frame * renderer.max_inflight_frames();
        let index_buffer = UniqueBuffer::new(
            renderer,
            BufferUsageFlags::INDEX_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            index_buffer_size as DeviceSize,
        )?;

        let ubo_bytes_one_frame = VulkanRenderer::aligned_size_of_type::<Uniform>(
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
        );

        let uniform_buffer = UniqueBuffer::new(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            ubo_bytes_one_frame * renderer.max_inflight_frames() as DeviceSize,
        )?;

        let mut imgui = Self::init_imgui(window);
        let font_atlas_image = imgui.fonts().build_alpha8_texture();

        let img_pixels = [ImageCopySource {
            src: font_atlas_image.data.as_ptr(),
            bytes: (font_atlas_image.width * font_atlas_image.height) as DeviceSize,
        }];

        let font_atlas_image = UniqueImage::new(
            renderer,
            ImageCreateInfo::builder()
                .usage(ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST)
                .format(Format::R8_UNORM)
                .initial_layout(ImageLayout::UNDEFINED)
                .image_type(ImageType::TYPE_2D)
                .tiling(ImageTiling::OPTIMAL)
                .array_layers(1)
                .mip_levels(1)
                .extent(Extent3D {
                    width: font_atlas_image.width,
                    height: font_atlas_image.height,
                    depth: 1,
                })
                .sharing_mode(SharingMode::EXCLUSIVE)
                .samples(SampleCountFlags::TYPE_1)
                .build(),
            &img_pixels,
        )?;

        let font_atlas_imageview = UniqueImageView::new(
            renderer,
            &ImageViewCreateInfo::builder()
                .format(Format::R8_UNORM)
                .image(font_atlas_image.image)
                .view_type(ImageViewType::TYPE_2D)
                .components(ComponentMapping::default())
                .subresource_range(
                    ImageSubresourceRange::builder()
                        .aspect_mask(ImageAspectFlags::COLOR)
                        .layer_count(1)
                        .level_count(1)
                        .build(),
                ),
        )?;

        let sampler = UniqueSampler::new(
            renderer.graphics_device(),
            &SamplerCreateInfo::builder()
                .mag_filter(Filter::LINEAR)
                .min_filter(Filter::LINEAR)
                .mipmap_mode(SamplerMipmapMode::LINEAR)
                .border_color(BorderColor::INT_OPAQUE_BLACK)
                .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(1f32)
                .build(),
        )?;

        let (pipeline_layout, descriptor_set_layouts) = GraphicsPipelineLayoutBuilder::new()
            .add_binding(
                DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .stage_flags(ShaderStageFlags::VERTEX)
                    .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .descriptor_count(1)
                    .build(),
            )
            .add_binding(
                DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .stage_flags(ShaderStageFlags::FRAGMENT)
                    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .build(),
            )
            .build(renderer.graphics_device())?;

        let descriptor_sets = unsafe {
            renderer.graphics_device().allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(renderer.descriptor_pool())
                    .set_layouts(&descriptor_set_layouts)
                    .build(),
            )
        }
        .map_err(|e| error!("Failed to allocate descriptor sets: {}", e))
        .ok()?;

        assert!(descriptor_sets.len() == 1);
        imgui.fonts().tex_id = TextureId::new(descriptor_sets[0].as_raw() as usize);

        let ds_buffer_info = [DescriptorBufferInfo::builder()
            .range(size_of::<Uniform>() as DeviceSize)
            .offset(0)
            .buffer(uniform_buffer.buffer)
            .build()];

        let ds_image_info = [DescriptorImageInfo::builder()
            .sampler(sampler.sampler)
            .image_view(font_atlas_imageview.view)
            .image_layout(ImageLayout::READ_ONLY_OPTIMAL)
            .build()];

        let wds = [
            WriteDescriptorSet::builder()
                .dst_binding(0)
                .dst_set(descriptor_sets[0])
                .dst_array_element(0)
                .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&ds_buffer_info)
                .build(),
            WriteDescriptorSet::builder()
                .dst_binding(1)
                .dst_array_element(0)
                .image_info(&ds_image_info)
                .dst_set(descriptor_sets[0])
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .build(),
        ];

        unsafe {
            renderer.graphics_device().update_descriptor_sets(&wds, &[]);
        }

        info!("Creating ui graphics pipeline");

        let pipeline = GraphicsPipelineBuilder::new()
            .add_vertex_input_attribute_description(
                VertexInputAttributeDescription::builder()
                    .location(0)
                    .binding(0)
                    .format(Format::R32G32_SFLOAT)
                    .offset(offset_of!(UiVertex, pos) as u32)
                    .build(),
            )
            .add_vertex_input_attribute_description(
                VertexInputAttributeDescription::builder()
                    .location(1)
                    .binding(0)
                    .format(Format::R32G32_SFLOAT)
                    .offset(offset_of!(UiVertex, uv) as u32)
                    .build(),
            )
            .add_vertex_input_attribute_description(
                VertexInputAttributeDescription::builder()
                    .location(2)
                    .binding(0)
                    .format(Format::R8G8B8A8_UNORM)
                    .offset(offset_of!(UiVertex, col) as u32)
                    .build(),
            )
            .set_input_assembly_state(PrimitiveTopology::TRIANGLE_LIST, false)
            .add_vertex_input_attribute_binding(
                VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(size_of::<UiVertex>() as u32)
                    .input_rate(VertexInputRate::VERTEX)
                    .build(),
            )
            .add_shader_stage(ShaderModuleDescription {
                stage: ShaderStageFlags::VERTEX,
                source: ShaderModuleSource::File(Path::new("data/shaders/ui.vert.spv")),
                entry_point: "main",
            })
            .add_shader_stage(ShaderModuleDescription {
                stage: ShaderStageFlags::FRAGMENT,
                source: ShaderModuleSource::File(Path::new("data/shaders/ui.frag.spv")),
                entry_point: "main",
            })
            .set_rasterization_state(
                PipelineRasterizationStateCreateInfo::builder()
                    .cull_mode(CullModeFlags::NONE)
                    .front_face(FrontFace::COUNTER_CLOCKWISE)
                    .line_width(1f32)
                    .polygon_mode(PolygonMode::FILL)
                    .build(),
            )
            .set_depth_test(false)
            .set_colorblend_attachment(
                0,
                PipelineColorBlendAttachmentState::builder()
                    .blend_enable(true)
                    .color_blend_op(BlendOp::ADD)
                    .alpha_blend_op(BlendOp::ADD)
                    .src_color_blend_factor(BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .dst_alpha_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .src_alpha_blend_factor(BlendFactor::ONE)
                    .color_write_mask(ColorComponentFlags::RGBA)
                    .build(),
            )
            .add_dynamic_state(DynamicState::VIEWPORT)
            .add_dynamic_state(DynamicState::SCISSOR)
            .build(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                pipeline_layout,
                descriptor_set_layouts,
                renderer.renderpass(),
                0,
            )?;

        info!("UI backend created ...");
        let win_size = window.get_size();
        let fb_size = window.get_framebuffer_size();

        Some(UiBackend {
            imgui: RefCell::new(imgui),
            vertex_bytes_one_frame: vertex_bytes_one_frame as u64,
            index_bytes_one_frame: index_bytes_one_frame as u64,
            ubo_bytes_one_frame,
            window_size: Cell::new(I32Vec2::new(win_size.0, win_size.1)),
            framebuffer_size: Cell::new(I32Vec2::new(fb_size.0, fb_size.1)),
            uniform_buffer,
            sampler,
            pipeline,
            descriptor_set: descriptor_sets[0],
            vertex_buffer,
            index_buffer,
            font_atlas_image,
            font_atlas_imageview,
        })
    }

    pub fn draw_frame(&self, draw_context: DrawContext) {
        let mut ui_context = self.imgui.borrow_mut();

        let draw_data = ui_context.render();
        assert!(draw_data.total_vtx_count < Self::MAX_VERTICES as i32);
        assert!(draw_data.total_idx_count < Self::MAX_INDICES as i32);

        let fb_width = (draw_data.display_size[0] * draw_data.framebuffer_scale[0]) as i32;
        let fb_height = (draw_data.display_size[1] * draw_data.framebuffer_scale[1]) as i32;
        if fb_width <= 0 || fb_height <= 0 {
            return;
        }

        if draw_data.total_vtx_count < 1 || draw_data.total_idx_count < 1 {
            return;
        }

        //
        // Push vertices + indices 2 GPU
        {
            let vertex_buffer_mapping = ScopedBufferMapping::create(
                draw_context.renderer,
                &self.vertex_buffer,
                self.vertex_bytes_one_frame,
                self.vertex_bytes_one_frame * draw_context.frame_id as u64,
            )
            .expect("Failed to map UI vertex buffer");

            let index_buffer_mapping = ScopedBufferMapping::create(
                draw_context.renderer,
                &self.index_buffer,
                self.index_bytes_one_frame,
                self.index_bytes_one_frame * draw_context.frame_id as u64,
            )
            .expect("Failed to map UI index buffer");

            let _ = draw_data.draw_lists().fold(
                (0isize, 0isize),
                |(vtx_offset, idx_offset), draw_list| {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            draw_list.vtx_buffer().as_ptr(),
                            (vertex_buffer_mapping.memptr() as *mut UiVertex).offset(vtx_offset),
                            draw_list.vtx_buffer().len(),
                        );

                        std::ptr::copy_nonoverlapping(
                            draw_list.idx_buffer().as_ptr(),
                            (index_buffer_mapping.memptr() as *mut UiIndex).offset(idx_offset),
                            draw_list.idx_buffer().len(),
                        );
                    }

                    (
                        vtx_offset + draw_list.vtx_buffer().len() as isize,
                        idx_offset + draw_list.idx_buffer().len() as isize,
                    )
                },
            );
        }

        unsafe {
            draw_context.graphics_device.cmd_bind_pipeline(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );

            let vertex_buffers = [self.vertex_buffer.buffer];
            let vertex_buffer_offsets =
                [(self.vertex_bytes_one_frame * draw_context.frame_id as u64) as DeviceSize];

            draw_context.graphics_device.cmd_bind_vertex_buffers(
                draw_context.cmd_buff,
                0,
                &vertex_buffers,
                &vertex_buffer_offsets,
            );

            draw_context.graphics_device.cmd_bind_index_buffer(
                draw_context.cmd_buff,
                self.index_buffer.buffer,
                (self.index_bytes_one_frame * draw_context.frame_id as u64) as DeviceSize,
                IndexType::UINT16,
            );

            let viewports = [draw_context.viewport];

            draw_context
                .graphics_device
                .cmd_set_viewport(draw_context.cmd_buff, 0, &viewports);

            let scissors = [draw_context.scissor];

            let scale = [
                2f32 / draw_data.display_size[0],
                2f32 / draw_data.display_size[1],
            ];

            let translate = [
                -1f32 - draw_data.display_pos[0] * scale[0],
                -1f32 - draw_data.display_pos[1] * scale[1],
            ];

            let transform = [
                scale[0],
                0.0f32,
                0.0f32,
                0.0f32,
                0.0f32,
                scale[1],
                0.0f32,
                0.0f32,
                0.0f32,
                0.0f32,
                1.0f32,
                0.0f32,
                translate[0],
                translate[1],
                0.0f32,
                1.0f32,
            ];

            let transform_gpu = std::slice::from_raw_parts(
                transform.as_ptr() as *const u8,
                transform.len() * size_of::<f32>(),
            );

            //
            // push transform
            {
                ScopedBufferMapping::create(
                    draw_context.renderer,
                    &self.uniform_buffer,
                    size_of::<Uniform>() as DeviceSize,
                    self.ubo_bytes_one_frame * draw_context.frame_id as DeviceSize,
                )
                .map(|mapping| {
                    std::ptr::copy_nonoverlapping(
                        transform_gpu.as_ptr(),
                        mapping.memptr() as *mut u8,
                        transform_gpu.len(),
                    );
                });
            }

            let descriptor_sets = [self.descriptor_set];
            let dynamic_offsets = [self.ubo_bytes_one_frame as u32 * draw_context.frame_id];

            draw_context.graphics_device.cmd_bind_descriptor_sets(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &descriptor_sets,
                &dynamic_offsets,
            );

            //
            // Will project scissor/clipping rectangles into framebuffer space
            let clip_off = draw_data.display_pos;
            let clip_scale = draw_data.framebuffer_scale;

            let _ = draw_data.draw_lists().fold(
                (0u32, 0u32),
                |(vertex_offset, index_offset), draw_list| {
                    for draw_cmd in draw_list.commands() {
                        match draw_cmd {
                            DrawCmd::Elements { count, cmd_params } => {
                                let mut clip_min = [
                                    (cmd_params.clip_rect[0] - clip_off[0]) * clip_scale[0],
                                    (cmd_params.clip_rect[1] - clip_off[1]) * clip_scale[1],
                                ];
                                let mut clip_max = [
                                    (cmd_params.clip_rect[2] - clip_off[0]) * clip_scale[0],
                                    (cmd_params.clip_rect[3] - clip_off[1]) * clip_scale[1],
                                ];
                                //
                                // Clamp to viewport as vkCmdSetScissor() won't accept values that are off bounds
                                if clip_min[0] < 0f32 {
                                    clip_min[0] = 0f32;
                                }

                                if clip_min[1] < 0f32 {
                                    clip_min[1] = 0f32;
                                }

                                if clip_max[0] > fb_width as f32 {
                                    clip_max[0] = fb_width as f32;
                                }

                                if clip_max[1] > fb_height as f32 {
                                    clip_max[1] = fb_height as f32;
                                }

                                if clip_max[0] <= clip_min[0] || clip_max[1] <= clip_min[1] {
                                    continue;
                                }

                                let scissor = [Rect2D {
                                    offset: Offset2D {
                                        x: clip_min[0] as i32,
                                        y: clip_min[1] as i32,
                                    },
                                    extent: Extent2D {
                                        width: (clip_max[0] - clip_min[0]).abs() as u32,
                                        height: (clip_max[1] - clip_min[1]).abs() as u32,
                                    },
                                }];

                                draw_context.graphics_device.cmd_set_scissor(
                                    draw_context.cmd_buff,
                                    0,
                                    &scissor,
                                );
                                draw_context.graphics_device.cmd_draw_indexed(
                                    draw_context.cmd_buff,
                                    count as u32,
                                    1,
                                    vertex_offset + cmd_params.idx_offset as u32,
                                    index_offset as i32 + cmd_params.vtx_offset as i32,
                                    0,
                                );
                            }
                            DrawCmd::ResetRenderState => info!("reset render state"),
                            _ => {}
                        }
                    }

                    (
                        vertex_offset + draw_list.vtx_buffer().len() as u32,
                        index_offset + draw_list.idx_buffer().len() as u32,
                    )
                },
            );
        }
    }

    fn init_imgui(window: &glfw::Window) -> imgui::Context {
        let mut imgui = imgui::Context::create();
        let io = imgui.io_mut();
        use glfw::Key as VirtualKeyCode;
        use imgui::Key;

        io[Key::Tab] = VirtualKeyCode::Tab as _;
        io[Key::LeftArrow] = VirtualKeyCode::Left as _;
        io[Key::RightArrow] = VirtualKeyCode::Right as _;
        io[Key::UpArrow] = VirtualKeyCode::Up as _;
        io[Key::DownArrow] = VirtualKeyCode::Down as _;
        io[Key::PageUp] = VirtualKeyCode::PageUp as _;
        io[Key::PageDown] = VirtualKeyCode::PageDown as _;
        io[Key::Home] = VirtualKeyCode::Home as _;
        io[Key::End] = VirtualKeyCode::End as _;
        io[Key::Insert] = VirtualKeyCode::Insert as _;
        io[Key::Delete] = VirtualKeyCode::Delete as _;
        io[Key::Backspace] = VirtualKeyCode::Backspace as _;
        io[Key::Space] = VirtualKeyCode::Space as _;
        io[Key::Enter] = VirtualKeyCode::Enter as _;
        io[Key::Escape] = VirtualKeyCode::Escape as _;
        io[Key::KeyPadEnter] = VirtualKeyCode::KpEnter as _;
        io[Key::A] = VirtualKeyCode::A as _;
        io[Key::C] = VirtualKeyCode::C as _;
        io[Key::V] = VirtualKeyCode::V as _;
        io[Key::X] = VirtualKeyCode::X as _;
        io[Key::Y] = VirtualKeyCode::Y as _;
        io[Key::Z] = VirtualKeyCode::Z as _;

        let (win_width, win_height) = window.get_size();
        io.display_size = [win_width as f32, win_height as f32];
        let (fb_width, fb_height) = window.get_framebuffer_size();
        io.display_framebuffer_scale = [
            fb_width as f32 / io.display_size[0],
            fb_height as f32 / io.display_size[1],
        ];

        imgui
    }

    pub fn handle_event(&self, event: &glfw::WindowEvent) {
        let mut context = self.imgui.borrow_mut();
        let mut io = context.io_mut();

        match *event {
            WindowEvent::FramebufferSize(width, height) => {
                self.framebuffer_size.set(I32Vec2::new(width, height));

                let winsize = self.window_size.get();
                let fbsize = self.framebuffer_size.get();

                if winsize.x > 0 && winsize.y > 0 {
                    io.display_framebuffer_scale = [
                        fbsize.x as f32 / winsize.x as f32,
                        fbsize.y as f32 / winsize.y as f32,
                    ];
                }
            }

            WindowEvent::Size(width, height) => {
                self.window_size.set(I32Vec2::new(width, height));

                let winsize = self.window_size.get();
                let fbsize = self.framebuffer_size.get();

                if winsize.x > 0 && winsize.y > 0 {
                    io.display_framebuffer_scale = [
                        fbsize.x as f32 / winsize.x as f32,
                        fbsize.y as f32 / winsize.y as f32,
                    ];
                }
            }

            WindowEvent::CursorPos(xpos, ypos) => {
                io.mouse_pos = [xpos as f32, ypos as f32];
            }

            WindowEvent::Scroll(xoffset, yoffset) => {
                io.mouse_wheel_h = yoffset as f32;
                io.mouse_wheel = xoffset as f32;
            }

            WindowEvent::Focus(focused) => io.app_focus_lost = !focused,

            WindowEvent::Key(key, _, action, modifiers) => {
                let pressed = action == Action::Press;
                let imguy_key = ImguiGlfwData::glfw_key2imgui_hey(key);
                io.keys_down[imguy_key as usize] = pressed;

                match modifiers {
                    Modifiers::Shift => io.key_shift = pressed,
                    Modifiers::Alt => io.key_alt = pressed,
                    Modifiers::Control => io.key_ctrl = pressed,
                    Modifiers::Super => io.key_super = pressed,
                    _ => {}
                }
            }

            WindowEvent::Char(ch) => {
                if ch != '\u{7f}' {
                    io.add_input_character(ch)
                }
            }

            WindowEvent::MouseButton(button, action, _modifiers) => {
                let pressed = action == Action::Press;
                match button {
                    MouseButton::Button1 => io.mouse_down[0] = pressed,
                    MouseButton::Button2 => io.mouse_down[1] = pressed,
                    MouseButton::Button3 => io.mouse_down[2] = pressed,
                    _ => {}
                }
            }

            _ => {}
        }
    }

    pub fn new_frame(&self) -> RefMut<imgui::Ui> {
        let ctx = self.imgui.borrow_mut();
        RefMut::map(ctx, |imgui_ctx| imgui_ctx.new_frame())
    }
}
