use ash::vk::{
    BlendFactor, BlendOp, BorderColor, BufferUsageFlags, ColorComponentFlags, ComponentMapping,
    CullModeFlags, DescriptorBufferInfo, DescriptorImageInfo, DescriptorSet,
    DescriptorSetAllocateInfo, DescriptorSetLayoutBinding, DescriptorType, DeviceSize,
    DynamicState, Extent2D, Extent3D, Filter, Format, FrontFace, Handle, ImageAspectFlags,
    ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags,
    ImageViewCreateInfo, ImageViewType, IndexType, MemoryPropertyFlags, Offset2D,
    PipelineBindPoint, PipelineColorBlendAttachmentState, PipelineRasterizationStateCreateInfo,
    PolygonMode, PrimitiveTopology, Rect2D, SampleCountFlags, SamplerAddressMode,
    SamplerCreateInfo, SamplerMipmapMode, ShaderStageFlags, SharingMode,
    VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
    WriteDescriptorSet,
};
use log::{error, info};
use memoffset::offset_of;

use imgui::{self, BackendFlags, DrawCmd, FontConfig, FontSource, Io, Key};

use winit::{
    dpi::{LogicalPosition, LogicalSize},
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, TouchPhase,
        VirtualKeyCode, WindowEvent,
    },
    window::{CursorIcon as MouseCursor, Window},
};

use crate::{
    draw_context::FrameRenderContext,
    vk_renderer::{
        GraphicsPipelineBuilder, GraphicsPipelineLayoutBuilder, ImageCopySource,
        ScopedBufferMapping, ShaderModuleDescription, ShaderModuleSource, UniqueBuffer,
        UniqueGraphicsPipeline, UniqueImage, UniqueImageView, UniqueSampler, VulkanRenderer,
    },
};

use std::{
    mem::size_of,
    path::Path,
};

type UiVertex = imgui::DrawVert;
type UiIndex = imgui::DrawIdx;

#[repr(C)]
struct Uniform {
    world_view_proj: nalgebra_glm::Mat4,
}

/// Parts adapted from imgui-winit-support example

/// winit backend platform state
#[derive(Debug)]
pub struct WinitPlatform {
    hidpi_mode: ActiveHiDpiMode,
    hidpi_factor: f64,
    cursor_cache: Option<CursorSettings>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct CursorSettings {
    cursor: Option<imgui::MouseCursor>,
    draw_cursor: bool,
}

fn to_winit_cursor(cursor: imgui::MouseCursor) -> MouseCursor {
    match cursor {
        imgui::MouseCursor::Arrow => MouseCursor::Default,
        imgui::MouseCursor::TextInput => MouseCursor::Text,
        imgui::MouseCursor::ResizeAll => MouseCursor::Move,
        imgui::MouseCursor::ResizeNS => MouseCursor::NsResize,
        imgui::MouseCursor::ResizeEW => MouseCursor::EwResize,
        imgui::MouseCursor::ResizeNESW => MouseCursor::NeswResize,
        imgui::MouseCursor::ResizeNWSE => MouseCursor::NwseResize,
        imgui::MouseCursor::Hand => MouseCursor::Hand,
        imgui::MouseCursor::NotAllowed => MouseCursor::NotAllowed,
    }
}

impl CursorSettings {
    fn apply(&self, window: &Window) {
        match self.cursor {
            Some(mouse_cursor) if !self.draw_cursor => {
                window.set_cursor_visible(true);
                window.set_cursor_icon(to_winit_cursor(mouse_cursor));
            }
            _ => window.set_cursor_visible(false),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum ActiveHiDpiMode {
    Default,
    Rounded,
    Locked,
}

/// DPI factor handling mode.
///
/// Applications that use imgui-rs might want to customize the used DPI factor and not use
/// directly the value coming from winit.
///
/// **Note: if you use a mode other than default and the DPI factor is adjusted, winit and imgui-rs
/// will use different logical coordinates, so be careful if you pass around logical size or
/// position values.**
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum HiDpiMode {
    /// The DPI factor from winit is used directly without adjustment
    Default,
    /// The DPI factor from winit is rounded to an integer value.
    ///
    /// This prevents the user interface from becoming blurry with non-integer scaling.
    Rounded,
    /// The DPI factor from winit is ignored, and the included value is used instead.
    ///
    /// This is useful if you want to force some DPI factor (e.g. 1.0) and not care about the value
    /// coming from winit.
    Locked(f64),
}

impl HiDpiMode {
    fn apply(&self, hidpi_factor: f64) -> (ActiveHiDpiMode, f64) {
        match *self {
            HiDpiMode::Default => (ActiveHiDpiMode::Default, hidpi_factor),
            HiDpiMode::Rounded => (ActiveHiDpiMode::Rounded, hidpi_factor.round()),
            HiDpiMode::Locked(value) => (ActiveHiDpiMode::Locked, value),
        }
    }
}

fn to_imgui_mouse_button(button: MouseButton) -> Option<imgui::MouseButton> {
    match button {
        MouseButton::Left | MouseButton::Other(0) => Some(imgui::MouseButton::Left),
        MouseButton::Right | MouseButton::Other(1) => Some(imgui::MouseButton::Right),
        MouseButton::Middle | MouseButton::Other(2) => Some(imgui::MouseButton::Middle),
        MouseButton::Other(3) => Some(imgui::MouseButton::Extra1),
        MouseButton::Other(4) => Some(imgui::MouseButton::Extra2),
        _ => None,
    }
}

fn to_imgui_key(keycode: VirtualKeyCode) -> Option<Key> {
    match keycode {
        VirtualKeyCode::Tab => Some(Key::Tab),
        VirtualKeyCode::Left => Some(Key::LeftArrow),
        VirtualKeyCode::Right => Some(Key::RightArrow),
        VirtualKeyCode::Up => Some(Key::UpArrow),
        VirtualKeyCode::Down => Some(Key::DownArrow),
        VirtualKeyCode::PageUp => Some(Key::PageUp),
        VirtualKeyCode::PageDown => Some(Key::PageDown),
        VirtualKeyCode::Home => Some(Key::Home),
        VirtualKeyCode::End => Some(Key::End),
        VirtualKeyCode::Insert => Some(Key::Insert),
        VirtualKeyCode::Delete => Some(Key::Delete),
        VirtualKeyCode::Back => Some(Key::Backspace),
        VirtualKeyCode::Space => Some(Key::Space),
        VirtualKeyCode::Return => Some(Key::Enter),
        VirtualKeyCode::Escape => Some(Key::Escape),
        VirtualKeyCode::LControl => Some(Key::LeftCtrl),
        VirtualKeyCode::LShift => Some(Key::LeftShift),
        VirtualKeyCode::LAlt => Some(Key::LeftAlt),
        VirtualKeyCode::LWin => Some(Key::LeftSuper),
        VirtualKeyCode::RControl => Some(Key::RightCtrl),
        VirtualKeyCode::RShift => Some(Key::RightShift),
        VirtualKeyCode::RAlt => Some(Key::RightAlt),
        VirtualKeyCode::RWin => Some(Key::RightSuper),
        //VirtualKeyCode::Menu => Some(Key::Menu), // TODO: find out if there is a Menu key in winit
        VirtualKeyCode::Key0 => Some(Key::Alpha0),
        VirtualKeyCode::Key1 => Some(Key::Alpha1),
        VirtualKeyCode::Key2 => Some(Key::Alpha2),
        VirtualKeyCode::Key3 => Some(Key::Alpha3),
        VirtualKeyCode::Key4 => Some(Key::Alpha4),
        VirtualKeyCode::Key5 => Some(Key::Alpha5),
        VirtualKeyCode::Key6 => Some(Key::Alpha6),
        VirtualKeyCode::Key7 => Some(Key::Alpha7),
        VirtualKeyCode::Key8 => Some(Key::Alpha8),
        VirtualKeyCode::Key9 => Some(Key::Alpha9),
        VirtualKeyCode::A => Some(Key::A),
        VirtualKeyCode::B => Some(Key::B),
        VirtualKeyCode::C => Some(Key::C),
        VirtualKeyCode::D => Some(Key::D),
        VirtualKeyCode::E => Some(Key::E),
        VirtualKeyCode::F => Some(Key::F),
        VirtualKeyCode::G => Some(Key::G),
        VirtualKeyCode::H => Some(Key::H),
        VirtualKeyCode::I => Some(Key::I),
        VirtualKeyCode::J => Some(Key::J),
        VirtualKeyCode::K => Some(Key::K),
        VirtualKeyCode::L => Some(Key::L),
        VirtualKeyCode::M => Some(Key::M),
        VirtualKeyCode::N => Some(Key::N),
        VirtualKeyCode::O => Some(Key::O),
        VirtualKeyCode::P => Some(Key::P),
        VirtualKeyCode::Q => Some(Key::Q),
        VirtualKeyCode::R => Some(Key::R),
        VirtualKeyCode::S => Some(Key::S),
        VirtualKeyCode::T => Some(Key::T),
        VirtualKeyCode::U => Some(Key::U),
        VirtualKeyCode::V => Some(Key::V),
        VirtualKeyCode::W => Some(Key::W),
        VirtualKeyCode::X => Some(Key::X),
        VirtualKeyCode::Y => Some(Key::Y),
        VirtualKeyCode::Z => Some(Key::Z),
        VirtualKeyCode::F1 => Some(Key::F1),
        VirtualKeyCode::F2 => Some(Key::F2),
        VirtualKeyCode::F3 => Some(Key::F3),
        VirtualKeyCode::F4 => Some(Key::F4),
        VirtualKeyCode::F5 => Some(Key::F5),
        VirtualKeyCode::F6 => Some(Key::F6),
        VirtualKeyCode::F7 => Some(Key::F7),
        VirtualKeyCode::F8 => Some(Key::F8),
        VirtualKeyCode::F9 => Some(Key::F9),
        VirtualKeyCode::F10 => Some(Key::F10),
        VirtualKeyCode::F11 => Some(Key::F11),
        VirtualKeyCode::F12 => Some(Key::F12),
        VirtualKeyCode::Apostrophe => Some(Key::Apostrophe),
        VirtualKeyCode::Comma => Some(Key::Comma),
        VirtualKeyCode::Minus => Some(Key::Minus),
        VirtualKeyCode::Period => Some(Key::Period),
        VirtualKeyCode::Slash => Some(Key::Slash),
        VirtualKeyCode::Semicolon => Some(Key::Semicolon),
        VirtualKeyCode::Equals => Some(Key::Equal),
        VirtualKeyCode::LBracket => Some(Key::LeftBracket),
        VirtualKeyCode::Backslash => Some(Key::Backslash),
        VirtualKeyCode::RBracket => Some(Key::RightBracket),
        VirtualKeyCode::Grave => Some(Key::GraveAccent),
        VirtualKeyCode::Capital => Some(Key::CapsLock),
        VirtualKeyCode::Scroll => Some(Key::ScrollLock),
        VirtualKeyCode::Numlock => Some(Key::NumLock),
        VirtualKeyCode::Snapshot => Some(Key::PrintScreen),
        VirtualKeyCode::Pause => Some(Key::Pause),
        VirtualKeyCode::Numpad0 => Some(Key::Keypad0),
        VirtualKeyCode::Numpad1 => Some(Key::Keypad1),
        VirtualKeyCode::Numpad2 => Some(Key::Keypad2),
        VirtualKeyCode::Numpad3 => Some(Key::Keypad3),
        VirtualKeyCode::Numpad4 => Some(Key::Keypad4),
        VirtualKeyCode::Numpad5 => Some(Key::Keypad5),
        VirtualKeyCode::Numpad6 => Some(Key::Keypad6),
        VirtualKeyCode::Numpad7 => Some(Key::Keypad7),
        VirtualKeyCode::Numpad8 => Some(Key::Keypad8),
        VirtualKeyCode::Numpad9 => Some(Key::Keypad9),
        VirtualKeyCode::NumpadDecimal => Some(Key::KeypadDecimal),
        VirtualKeyCode::NumpadDivide => Some(Key::KeypadDivide),
        VirtualKeyCode::NumpadMultiply => Some(Key::KeypadMultiply),
        VirtualKeyCode::NumpadSubtract => Some(Key::KeypadSubtract),
        VirtualKeyCode::NumpadAdd => Some(Key::KeypadAdd),
        VirtualKeyCode::NumpadEnter => Some(Key::KeypadEnter),
        VirtualKeyCode::NumpadEquals => Some(Key::KeypadEqual),
        _ => None,
    }
}

fn translate_winit_key(key: winit::event::VirtualKeyCode) -> imgui::Key {
    match key {
        VirtualKeyCode::Tab => imgui::Key::Tab,
        VirtualKeyCode::Left => imgui::Key::LeftArrow,
        VirtualKeyCode::Right => imgui::Key::RightArrow,
        VirtualKeyCode::Up => imgui::Key::UpArrow,
        VirtualKeyCode::Down => imgui::Key::DownArrow,
        VirtualKeyCode::PageUp => imgui::Key::PageUp,
        VirtualKeyCode::PageDown => imgui::Key::PageDown,
        VirtualKeyCode::Home => imgui::Key::Home,
        VirtualKeyCode::End => imgui::Key::End,
        VirtualKeyCode::Insert => imgui::Key::Insert,
        VirtualKeyCode::Delete => imgui::Key::Delete,
        VirtualKeyCode::Back => imgui::Key::Backspace,
        VirtualKeyCode::Space => imgui::Key::Space,
        VirtualKeyCode::Return => imgui::Key::Enter,
        VirtualKeyCode::Escape => imgui::Key::Escape,
        VirtualKeyCode::A => imgui::Key::A,
        VirtualKeyCode::C => imgui::Key::C,
        VirtualKeyCode::V => imgui::Key::V,
        VirtualKeyCode::X => imgui::Key::X,
        VirtualKeyCode::Y => imgui::Key::Y,
        VirtualKeyCode::Z => imgui::Key::Z,
        _ => imgui::Key::Escape,
    }
}

pub struct UiBackend {
    imgui: imgui::Context,
    vertex_bytes_one_frame: DeviceSize,
    index_bytes_one_frame: DeviceSize,
    ubo_bytes_one_frame: DeviceSize,
    uniform_buffer: UniqueBuffer,
    sampler: UniqueSampler,
    pipeline: UniqueGraphicsPipeline,
    descriptor_set: DescriptorSet,
    vertex_buffer: UniqueBuffer,
    index_buffer: UniqueBuffer,
    font_atlas_image: UniqueImage,
    font_atlas_imageview: UniqueImageView,
    platform: WinitPlatform,
}

impl UiBackend {
    const MAX_VERTICES: u32 = 8192;
    const MAX_INDICES: u32 = 16535;

    /// Scales a logical size coming from winit using the current DPI mode.
    ///
    /// This utility function is useful if you are using a DPI mode other than default, and want
    /// your application to use the same logical coordinates as imgui-rs.
    pub fn scale_size_from_winit(
        platform: &WinitPlatform,
        window: &Window,
        logical_size: LogicalSize<f64>,
    ) -> LogicalSize<f64> {
        match platform.hidpi_mode {
            ActiveHiDpiMode::Default => logical_size,
            _ => logical_size
                .to_physical::<f64>(window.scale_factor())
                .to_logical(platform.hidpi_factor),
        }
    }

    /// Scales a logical position coming from winit using the current DPI mode.
    ///
    /// This utility function is useful if you are using a DPI mode other than default, and want
    /// your application to use the same logical coordinates as imgui-rs.
    pub fn scale_pos_from_winit(
        platform: &WinitPlatform,
        window: &Window,
        logical_pos: LogicalPosition<f64>,
    ) -> LogicalPosition<f64> {
        match platform.hidpi_mode {
            ActiveHiDpiMode::Default => logical_pos,
            _ => logical_pos
                .to_physical::<f64>(window.scale_factor())
                .to_logical(platform.hidpi_factor),
        }
    }

    /// Scales a logical position for winit using the current DPI mode.
    ///
    /// This utility function is useful if you are using a DPI mode other than default, and want
    /// your application to use the same logical coordinates as imgui-rs.
    pub fn scale_pos_for_winit(
        platform: &WinitPlatform,
        window: &Window,
        logical_pos: LogicalPosition<f64>,
    ) -> LogicalPosition<f64> {
        match platform.hidpi_mode {
            ActiveHiDpiMode::Default => logical_pos,
            _ => logical_pos
                .to_physical::<f64>(platform.hidpi_factor)
                .to_logical(window.scale_factor()),
        }
    }

    fn init_imgui() -> imgui::Context {
        let mut imgui = imgui::Context::create();
        let io = imgui.io_mut();

        io.backend_flags.insert(BackendFlags::HAS_MOUSE_CURSORS);
        io.backend_flags.insert(BackendFlags::HAS_SET_MOUSE_POS);
        imgui.set_platform_name(Some(format!("B5 {}", env!("CARGO_PKG_VERSION"))));

        imgui
    }

    pub fn new(renderer: &VulkanRenderer, window: &winit::window::Window) -> Option<UiBackend> {
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

        let mut imgui = Self::init_imgui();
        let hidpi_mode = HiDpiMode::Default;
        let (hidpi_mode, hidpi_factor) = hidpi_mode.apply(window.scale_factor());

        let platform = WinitPlatform {
            hidpi_mode,
            hidpi_factor,
            cursor_cache: None,
        };

        imgui.io_mut().display_framebuffer_scale = [hidpi_factor as f32, hidpi_factor as f32];
        let logical_size = window.inner_size().to_logical(hidpi_factor);
        let logical_size = Self::scale_size_from_winit(&platform, window, logical_size);
        imgui.io_mut().display_size = [logical_size.width as f32, logical_size.height as f32];

        let font_files = [
            "data/fonts/iosevka-ss03-regular.ttf",
            "data/fonts/iosevka-ss03-medium.ttf",
            "data/fonts/RobotoMono-Medium.ttf",
            "data/fonts/RobotoMono-Regular.ttf",
        ];

        if let Ok(mut font_file) = std::fs::File::open(font_files[0]) {
            let mut ttf_bytes = Vec::<u8>::new();
            use std::io::Read;

            if let Ok(_) = font_file.read_to_end(&mut ttf_bytes) {
                imgui.fonts().add_font(&[FontSource::TtfData {
                    data: &ttf_bytes,
                    size_pixels: 18f32,
                    config: Some(FontConfig {
                        oversample_h: 4,
                        oversample_v: 4,
                        rasterizer_multiply: 1.5f32,
                        ..FontConfig::default()
                    }),
                }]);
            }
        }

        let font_atlas_image = imgui.fonts().build_alpha8_texture();

        let img_pixels = [ImageCopySource {
            src: font_atlas_image.data.as_ptr(),
            bytes: (font_atlas_image.width * font_atlas_image.height) as DeviceSize,
        }];

        let copy_tex_work_pkg = renderer.create_work_package()?;

        let font_atlas_image = UniqueImage::with_data(
            renderer,
            &ImageCreateInfo::builder()
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
            &copy_tex_work_pkg,
        )?;

        renderer.push_work_package(copy_tex_work_pkg);

        let font_atlas_imageview = UniqueImageView::new(
            renderer.graphics_device(),
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

        info!("Creating ui graphics pipeline");

        let pipeline = GraphicsPipelineBuilder::new()
            .add_vertex_input_attribute_descriptions(&[
                VertexInputAttributeDescription::builder()
                    .location(0)
                    .binding(0)
                    .format(Format::R32G32_SFLOAT)
                    .offset(offset_of!(UiVertex, pos) as u32)
                    .build(),
                VertexInputAttributeDescription::builder()
                    .location(1)
                    .binding(0)
                    .format(Format::R32G32_SFLOAT)
                    .offset(offset_of!(UiVertex, uv) as u32)
                    .build(),
                VertexInputAttributeDescription::builder()
                    .location(2)
                    .binding(0)
                    .format(Format::R8G8B8A8_UNORM)
                    .offset(offset_of!(UiVertex, col) as u32)
                    .build(),
            ])
            .set_input_assembly_state(PrimitiveTopology::TRIANGLE_LIST, false)
            .add_vertex_input_attribute_binding(
                VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(size_of::<UiVertex>() as u32)
                    .input_rate(VertexInputRate::VERTEX)
                    .build(),
            )
            .shader_stages(&[
                ShaderModuleDescription {
                    stage: ShaderStageFlags::VERTEX,
                    source: ShaderModuleSource::File(Path::new("data/shaders/ui.vert.spv")),
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(Path::new("data/shaders/ui.frag.spv")),
                    entry_point: "main",
                },
            ])
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
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR])
            .build(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                GraphicsPipelineLayoutBuilder::new()
                    .set(
                        0,
                        &[
                            DescriptorSetLayoutBinding::builder()
                                .binding(0)
                                .stage_flags(ShaderStageFlags::VERTEX)
                                .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                                .descriptor_count(1)
                                .build(),
                            DescriptorSetLayoutBinding::builder()
                                .binding(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .build(),
                        ],
                    )
                    .build(renderer.graphics_device())?,
                renderer.renderpass(),
                0,
            )?;

        let descriptor_sets = unsafe {
            renderer.graphics_device().allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(renderer.descriptor_pool())
                    .set_layouts(pipeline.descriptor_layouts())
                    .build(),
            )
        }
        .map_err(|e| error!("Failed to allocate descriptor sets: {}", e))
        .ok()?;

        assert!(descriptor_sets.len() == 1);
        imgui.fonts().tex_id = imgui::TextureId::new(descriptor_sets[0].as_raw() as usize);

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

        Some(UiBackend {
            imgui,
            vertex_bytes_one_frame: vertex_bytes_one_frame as u64,
            index_bytes_one_frame: index_bytes_one_frame as u64,
            ubo_bytes_one_frame,
            uniform_buffer,
            sampler,
            pipeline,
            descriptor_set: descriptor_sets[0],
            vertex_buffer,
            index_buffer,
            font_atlas_image,
            font_atlas_imageview,
            platform,
        })
    }

    pub fn draw_frame(&mut self, frame_context: &FrameRenderContext) {
        let ui_context = &mut self.imgui;

        let draw_data = ui_context.render();
        assert!(draw_data.total_vtx_count < Self::MAX_VERTICES as i32);
        assert!(draw_data.total_idx_count < Self::MAX_INDICES as i32);

        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
        if !(fb_width > 0.0 && fb_height > 0.0) {
            return;
        }

        if draw_data.total_vtx_count < 1 || draw_data.total_idx_count < 1 {
            return;
        }

        //
        // Push vertices + indices 2 GPU
        {
            let vertex_buffer_mapping = ScopedBufferMapping::create(
                frame_context.renderer,
                &self.vertex_buffer,
                self.vertex_bytes_one_frame,
                self.vertex_bytes_one_frame * frame_context.frame_id as u64,
            )
            .expect("Failed to map UI vertex buffer");

            let index_buffer_mapping = ScopedBufferMapping::create(
                frame_context.renderer,
                &self.index_buffer,
                self.index_bytes_one_frame,
                self.index_bytes_one_frame * frame_context.frame_id as u64,
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

        let graphics_device = frame_context.renderer.graphics_device();

        unsafe {
            graphics_device.cmd_bind_pipeline(
                frame_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );

            let vertex_buffers = [self.vertex_buffer.buffer];
            let vertex_buffer_offsets =
                [(self.vertex_bytes_one_frame * frame_context.frame_id as u64) as DeviceSize];

            graphics_device.cmd_bind_vertex_buffers(
                frame_context.cmd_buff,
                0,
                &vertex_buffers,
                &vertex_buffer_offsets,
            );

            graphics_device.cmd_bind_index_buffer(
                frame_context.cmd_buff,
                self.index_buffer.buffer,
                (self.index_bytes_one_frame * frame_context.frame_id as u64) as DeviceSize,
                IndexType::UINT16,
            );

            let viewports = [frame_context.viewport];

            graphics_device.cmd_set_viewport(frame_context.cmd_buff, 0, &viewports);

            let _scissors = [frame_context.scissor];

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
                    frame_context.renderer,
                    &self.uniform_buffer,
                    size_of::<Uniform>() as DeviceSize,
                    self.ubo_bytes_one_frame * frame_context.frame_id as DeviceSize,
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
            let dynamic_offsets = [self.ubo_bytes_one_frame as u32 * frame_context.frame_id];

            graphics_device.cmd_bind_descriptor_sets(
                frame_context.cmd_buff,
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

                                graphics_device.cmd_set_scissor(
                                    frame_context.cmd_buff,
                                    0,
                                    &scissor,
                                );
                                graphics_device.cmd_draw_indexed(
                                    frame_context.cmd_buff,
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

    pub fn new_frame(&mut self) -> &mut imgui::Ui {
        self.imgui.new_frame()
    }

    fn handle_key_modifier(io: &mut Io, key: VirtualKeyCode, down: bool) {
        if key == VirtualKeyCode::LShift || key == VirtualKeyCode::RShift {
            io.add_key_event(imgui::Key::ModShift, down);
        } else if key == VirtualKeyCode::LControl || key == VirtualKeyCode::RControl {
            io.add_key_event(imgui::Key::ModCtrl, down);
        } else if key == VirtualKeyCode::LAlt || key == VirtualKeyCode::RAlt {
            io.add_key_event(imgui::Key::ModAlt, down);
        } else if key == VirtualKeyCode::LWin || key == VirtualKeyCode::RWin {
            io.add_key_event(imgui::Key::ModSuper, down);
        }
    }

    /// Handles a winit event.
    ///
    /// This function performs the following actions (depends on the event):
    ///
    /// * window size / dpi factor changes are applied
    /// * keyboard state is updated
    /// * mouse state is updated
    pub fn handle_event<T>(&mut self, window: &Window, event: &Event<T>) -> bool {
        match *event {
            Event::WindowEvent {
                window_id,
                ref event,
            } if window_id == window.id() => {
                self.handle_window_event(window, event);
            }
            // Track key release events outside our window. If we don't do this,
            // we might never see the release event if some other window gets focus.
            Event::DeviceEvent {
                event:
                    DeviceEvent::Key(KeyboardInput {
                        state: ElementState::Released,
                        virtual_keycode: Some(key),
                        ..
                    }),
                ..
            } => {
                if let Some(key) = to_imgui_key(key) {
                    self.imgui.io_mut().add_key_event(key, false);
                }
            }
            _ => (),
        }

        let (wants_keys, wants_mouse) = (
            self.imgui.io().want_capture_keyboard,
            self.imgui.io().want_capture_mouse,
        );

        wants_keys || wants_mouse
    }

    fn handle_window_event(&mut self, window: &Window, event: &WindowEvent) {
        match *event {
            WindowEvent::Resized(physical_size) => {
                let logical_size = physical_size.to_logical(window.scale_factor());
                let logical_size =
                    Self::scale_size_from_winit(&self.platform, window, logical_size);
                self.imgui.io_mut().display_size =
                    [logical_size.width as f32, logical_size.height as f32];
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                let hidpi_factor = match self.platform.hidpi_mode {
                    ActiveHiDpiMode::Default => scale_factor,
                    ActiveHiDpiMode::Rounded => scale_factor.round(),
                    _ => return,
                };
                // Mouse position needs to be changed while we still have both the old and the new
                // values
                if self.imgui.io().mouse_pos[0].is_finite()
                    && self.imgui.io().mouse_pos[1].is_finite()
                {
                    self.imgui.io_mut().mouse_pos = [
                        self.imgui.io().mouse_pos[0]
                            * (hidpi_factor / self.platform.hidpi_factor) as f32,
                        self.imgui.io().mouse_pos[1]
                            * (hidpi_factor / self.platform.hidpi_factor) as f32,
                    ];
                }

                self.platform.hidpi_factor = hidpi_factor;
                self.imgui.io_mut().display_framebuffer_scale =
                    [hidpi_factor as f32, hidpi_factor as f32];
                // Window size might change too if we are using DPI rounding
                let logical_size = window.inner_size().to_logical(scale_factor);
                let logical_size =
                    Self::scale_size_from_winit(&self.platform, window, logical_size);
                self.imgui.io_mut().display_size =
                    [logical_size.width as f32, logical_size.height as f32];
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                // We need to track modifiers separately because some system like macOS, will
                // not reliably send modifier states during certain events like ScreenCapture.
                // Gotta let the people show off their pretty imgui widgets!
                let io = self.imgui.io_mut();
                io.add_key_event(Key::ModShift, modifiers.shift());
                io.add_key_event(Key::ModCtrl, modifiers.ctrl());
                io.add_key_event(Key::ModAlt, modifiers.alt());
                io.add_key_event(Key::ModSuper, modifiers.logo());
            }
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(key),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;

                // We map both left and right ctrl to `ModCtrl`, etc.
                // imgui is told both "left control is pressed" and
                // "consider the control key is pressed". Allows
                // applications to use either general "ctrl" or a
                // specific key. Same applies to other modifiers.
                // https://github.com/ocornut/imgui/issues/5047
                Self::handle_key_modifier(self.imgui.io_mut(), key, pressed);

                // Add main key event
                if let Some(key) = to_imgui_key(key) {
                    self.imgui.io_mut().add_key_event(key, pressed);
                }
            }
            WindowEvent::ReceivedCharacter(ch) => {
                // Exclude the backspace key ('\u{7f}'). Otherwise we will insert this char and then
                // delete it.
                if ch != '\u{7f}' {
                    self.imgui.io_mut().add_input_character(ch);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = position.to_logical(window.scale_factor());
                let position = Self::scale_pos_from_winit(&self.platform, window, position);
                self.imgui
                    .io_mut()
                    .add_mouse_pos_event([position.x as f32, position.y as f32]);
            }
            WindowEvent::MouseWheel {
                delta,
                phase: TouchPhase::Moved,
                ..
            } => {
                use std::cmp::Ordering;
                let (h, v) = match delta {
                    MouseScrollDelta::LineDelta(h, v) => (h, v),
                    MouseScrollDelta::PixelDelta(pos) => {
                        let pos = pos.to_logical::<f64>(self.platform.hidpi_factor);
                        let h = match pos.x.partial_cmp(&0.0) {
                            Some(Ordering::Greater) => 1.0,
                            Some(Ordering::Less) => -1.0,
                            _ => 0.0,
                        };
                        let v = match pos.y.partial_cmp(&0.0) {
                            Some(Ordering::Greater) => 1.0,
                            Some(Ordering::Less) => -1.0,
                            _ => 0.0,
                        };
                        (h, v)
                    }
                };
                self.imgui.io_mut().add_mouse_wheel_event([h, v]);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(mb) = to_imgui_mouse_button(button) {
                    let pressed = state == ElementState::Pressed;
                    self.imgui.io_mut().add_mouse_button_event(mb, pressed);
                }
            }
            WindowEvent::Focused(newly_focused) => {
                if !newly_focused {
                    // Set focus-lost to avoid stuck keys (like 'alt'
                    // when alt-tabbing)
                    self.imgui.io_mut().app_focus_lost = true;
                }
            }
            _ => (),
        }
    }
}
