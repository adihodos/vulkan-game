use ash::vk::{
    BufferUsageFlags, CommandBuffer, ComponentMapping, CullModeFlags, DescriptorBufferInfo,
    DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayoutBinding, DescriptorType,
    DeviceSize, DynamicState, Extent2D, Extent3D, Format, FrontFace, ImageAspectFlags,
    ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags,
    ImageViewCreateFlags, ImageViewCreateInfo, ImageViewType, IndexType, MemoryPropertyFlags,
    MemoryType, Offset2D, PipelineBindPoint, PipelineRasterizationStateCreateInfo, PolygonMode,
    Rect2D, SampleCountFlags, ShaderStageFlags, SharingMode, VertexInputAttributeDescription,
    VertexInputBindingDescription, VertexInputRate, Viewport, WriteDescriptorSet,
};
use chrono::Duration;
use glfw::{Action, Context, Key};
use glm::{IVec2, Vec3};
use imgui::Condition;
use log::{debug, error, info, trace, warn};
use nalgebra_glm::{Mat4, Vec4};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fs::File,
    io::Write,
    mem::size_of,
    path::{Path, PathBuf},
    ptr::copy_nonoverlapping,
    sync::mpsc::Receiver,
    time::Instant,
};

mod app_config;
mod arcball_camera;
mod camera;
mod draw_context;
mod imported_geometry;
mod pbr;
mod resource_cache;
mod skybox;
mod starfury;
mod ui_backend;
mod vk_renderer;

use nalgebra_glm as glm;

use crate::{
    app_config::AppConfig,
    arcball_camera::ArcballCamera,
    camera::Camera,
    draw_context::DrawContext,
    imported_geometry::{GeometryVertex, ImportedGeometry},
    pbr::{PbrMaterial, PbrMaterialTextureCollection},
    resource_cache::ResourceHolder,
    skybox::Skybox,
    vk_renderer::{
        GraphicsPipelineBuilder, GraphicsPipelineLayoutBuilder, RendererWorkPackage,
        ScopedBufferMapping, ShaderModuleDescription, ShaderModuleSource, UniqueBuffer,
        UniqueGraphicsPipeline, UniqueImage, UniqueImageView, UniqueSampler, VulkanRenderer,
    },
};

#[repr(C)]
struct WireframeShaderUBO {
    transform: Mat4,
    color: Vec4,
}

#[derive(Copy, Clone, Debug)]
struct DrawOpts {
    wireframe_color: Vec4,
    draw_normals: bool,
    normals_color: Vec4,
}

struct OKurwaJebaneObject {
    draw_opts: RefCell<DrawOpts>,
    ubo_bytes_one_frame: DeviceSize,
    ubo: UniqueBuffer,
    resource_cache: ResourceHolder,
    pipeline: UniqueGraphicsPipeline,
    descriptor_sets: Vec<DescriptorSet>,
    skybox: Skybox,
}

impl OKurwaJebaneObject {
    pub fn new(renderer: &VulkanRenderer, app_cfg: &AppConfig) -> Option<OKurwaJebaneObject> {
        let resource_cache = ResourceHolder::create(renderer, app_cfg)?;

        let pipeline = GraphicsPipelineBuilder::new()
            .add_vertex_input_attribute_description(
                VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(0)
                    .location(0)
                    .format(Format::R32G32B32_SFLOAT)
                    .build(),
            )
            .add_vertex_input_attribute_binding(
                VertexInputBindingDescription::builder()
                    .stride(size_of::<GeometryVertex>() as u32)
                    .binding(0)
                    .input_rate(VertexInputRate::VERTEX)
                    .build(),
            )
            .set_rasterization_state(
                PipelineRasterizationStateCreateInfo::builder()
                    .cull_mode(CullModeFlags::BACK)
                    .front_face(FrontFace::COUNTER_CLOCKWISE)
                    .polygon_mode(PolygonMode::LINE)
                    .line_width(1f32)
                    .build(),
            )
            .add_shader_stage(ShaderModuleDescription {
                stage: ShaderStageFlags::VERTEX,
                source: ShaderModuleSource::File(Path::new("data/shaders/wireframe.vert.spv")),
                entry_point: "main",
            })
            .add_shader_stage(ShaderModuleDescription {
                stage: ShaderStageFlags::FRAGMENT,
                source: ShaderModuleSource::File(Path::new("data/shaders/wireframe.frag.spv")),
                entry_point: "main",
            })
            .add_dynamic_state(DynamicState::VIEWPORT)
            .add_dynamic_state(DynamicState::SCISSOR)
            .build(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                GraphicsPipelineLayoutBuilder::new()
                    .add_binding(
                        DescriptorSetLayoutBinding::builder()
                            .stage_flags(ShaderStageFlags::VERTEX)
                            .binding(0)
                            .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .descriptor_count(1)
                            .build(),
                    )
                    .build(renderer.graphics_device())?,
                renderer.renderpass(),
                0,
            )?;

        let ubo_bytes_one_frame = VulkanRenderer::aligned_size_of_type::<WireframeShaderUBO>(
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
        );

        let ubo = UniqueBuffer::new(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            ubo_bytes_one_frame * renderer.max_inflight_frames() as u64,
        )?;

        let descriptor_sets = renderer.allocate_descriptor_sets(&pipeline)?;

        let dbi = [DescriptorBufferInfo::builder()
            .buffer(ubo.buffer)
            .offset(0)
            .range(size_of::<WireframeShaderUBO>() as DeviceSize)
            .build()];

        let wds = [WriteDescriptorSet::builder()
            .dst_set(descriptor_sets[0])
            .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .dst_binding(0)
            .dst_array_element(0)
            .buffer_info(&dbi)
            .build()];

        unsafe { renderer.graphics_device().update_descriptor_sets(&wds, &[]) }

        Some(OKurwaJebaneObject {
            draw_opts: RefCell::new(DrawOpts {
                wireframe_color: Vec4::new(0f32, 1f32, 0f32, 1f32),
                draw_normals: false,
                normals_color: Vec4::new(1f32, 0f32, 0f32, 1f32),
            }),
            ubo_bytes_one_frame,
            ubo,
            resource_cache,
            pipeline,
            descriptor_sets,
            skybox: Skybox::create(renderer, &app_cfg.scene, &app_cfg.engine)?,
        })
    }

    pub fn draw(&self, draw_context: &DrawContext) {
        self.skybox.draw(draw_context);

        let device = draw_context.renderer.graphics_device();

        let viewports = [draw_context.viewport];
        let scisssors = [draw_context.scissor];

        let view_matrix = draw_context.camera.view_transform();

        let perspective = draw_context.projection;
        let ubo_data = WireframeShaderUBO {
            transform: perspective * view_matrix,
            color: self.draw_opts.borrow().wireframe_color,
        };

        ScopedBufferMapping::create(
            draw_context.renderer,
            &self.ubo,
            self.ubo_bytes_one_frame,
            self.ubo_bytes_one_frame * draw_context.frame_id as u64,
        )
        .map(|mapping| unsafe {
            copy_nonoverlapping(
                &ubo_data as *const _,
                mapping.memptr() as *mut WireframeShaderUBO,
                1,
            );
        });

        unsafe {
            device.cmd_bind_pipeline(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            device.cmd_set_viewport(draw_context.cmd_buff, 0, &viewports);
            device.cmd_set_scissor(draw_context.cmd_buff, 0, &scisssors);

            let sa23_handle = self.resource_cache.get_geometry_handle("sa23");
            let sa23_geom = self.resource_cache.get_renderable_geometry(sa23_handle);

            let vertex_buffers = [self.resource_cache.vertex_buffer()];
            let vertex_offsets = [0u64];
            device.cmd_bind_vertex_buffers(
                draw_context.cmd_buff,
                0,
                &vertex_buffers,
                &vertex_offsets,
            );
            device.cmd_bind_index_buffer(
                draw_context.cmd_buff,
                self.resource_cache.index_buffer(),
                0,
                IndexType::UINT32,
            );

            let ubo_offsets = [self.ubo_bytes_one_frame as u32 * draw_context.frame_id];
            device.cmd_bind_descriptor_sets(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &self.descriptor_sets,
                &ubo_offsets,
            );

            device.cmd_draw_indexed(
                draw_context.cmd_buff,
                sa23_geom.index_count,
                1,
                sa23_geom.index_offset,
                sa23_geom.vertex_offset as i32,
                0,
            );
        }
    }

    pub fn ui(&self, ui: &mut imgui::Ui) {
        let choices = ["test test this is 1", "test test this is 2"];
        ui.window("Hello world")
            .size([300.0, 110.0], Condition::FirstUseEver)
            .build(|| {
                let mut draw_opts = self.draw_opts.borrow_mut();
                let mut wf_color = [
                    draw_opts.wireframe_color.x,
                    draw_opts.wireframe_color.y,
                    draw_opts.wireframe_color.z,
                ];
                if ui.color_picker3("wireframe color", &mut wf_color) {
                    draw_opts.wireframe_color =
                        Vec4::new(wf_color[0], wf_color[1], wf_color[2], 1f32);
                }
            });
    }
}

struct BasicWindow {
    glfw: glfw::Glfw,
    window: glfw::Window,
    kurwa: OKurwaJebaneObject,
    ui: ui_backend::UiBackend,
    renderer: std::cell::RefCell<VulkanRenderer>,
    camera: ArcballCamera,
    fb_size: Cell<IVec2>,
}

impl BasicWindow {
    pub fn new(
        glfw: glfw::Glfw,
        window: glfw::Window,
        renderer: VulkanRenderer,
        app_cfg: &AppConfig,
    ) -> Option<BasicWindow> {
        renderer.begin_resource_loading();

        let kurwa = OKurwaJebaneObject::new(&renderer, app_cfg)?;
        let ui = ui_backend::UiBackend::new(&renderer, &window)?;

        renderer.wait_all_work_packages();
        renderer.wait_resources_loaded();
        info!("Resource loaded ...");

        let (width, height) = window.get_framebuffer_size();
        let fb_size = IVec2::new(width, height);

        Some(BasicWindow {
            glfw,
            window,
            kurwa,
            ui,
            renderer: RefCell::new(renderer),
            camera: ArcballCamera::new(Vec3::new(0f32, 0f32, 0f32), 0.1f32, fb_size),
            fb_size: Cell::new(fb_size),
        })
    }

    fn main_loop(&mut self, events: &Receiver<(f64, glfw::WindowEvent)>) {
        self.window.set_all_polling(true);

        while !self.window.should_close() {
            self.glfw.poll_events();

            let queued_events = glfw::flush_messages(events);

            for (_, event) in queued_events {
                self.handle_window_event(&event);
                self.ui.handle_event(&event);
            }

            self.do_ui();
            self.draw_frame();
        }

        self.renderer.borrow().wait_idle();
    }

    fn draw_frame(&self) {
        let renderer = self.renderer.borrow();

        renderer.begin_frame();
        {
            let fb_size = self.fb_size.get();

            let draw_context = DrawContext::create(
                &renderer,
                fb_size.x,
                fb_size.y,
                &self.camera,
                perspective(75f32, fb_size.x as f32 / fb_size.y as f32, 0.1f32, 5000f32),
            );

            self.kurwa.draw(&draw_context);
            self.ui.draw_frame(&draw_context);
        }
        renderer.end_frame();
    }

    fn do_ui(&self) {
        let mut ui = self.ui.new_frame();
        self.kurwa.ui(&mut ui);
    }

    fn handle_window_event(&mut self, event: &glfw::WindowEvent) {
        if let glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) = *event {
            self.window.set_should_close(true);
            return;
        }
        self.camera.input_event(event);
    }
}

fn main() {
    let logger = flexi_logger::Logger::with(
        flexi_logger::LogSpecification::builder()
            .default(flexi_logger::LevelFilter::Trace)
            .build(),
    )
    .adaptive_format_for_stderr(flexi_logger::AdaptiveFormat::Detailed)
    .start()
    .unwrap_or_else(|e| {
        panic!("Failed to start the logger {}", e);
    });

    let app_config = AppConfig::load();

    info!("uraaa this be info!");
    warn!("urraa! this be warn cyka!");
    error!("urrra! this be error pierdole!");
    trace!("urrraa ! this be trace blyat!");
    debug!("urraa! this be debug, kurwa jebane !");

    glfw::init(glfw::FAIL_ON_ERRORS)
        .and_then(|mut glfw| {
            glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
            glfw.window_hint(glfw::WindowHint::Decorated(false));
            let mut vidmode = glfw.with_primary_monitor(|_, pmon| pmon.map(|p| p.get_video_mode()));

            vidmode
                .take()
                .map(move |vidmode| vidmode.map(|v| (glfw, v)))
                .ok_or(glfw::InitError::Internal)
        })
        .ok()
        .flatten()
        .and_then(|(mut glfw, vidmode)| {
            glfw.create_window(
                vidmode.width,
                vidmode.height,
                "Vulkan + Rust + Babylon5",
                glfw::WindowMode::Windowed,
            )
            .map(move |(window, events)| (glfw, window, events))
        })
        .and_then(|(mut glfw, mut window, events)| {
            let renderer = VulkanRenderer::create(&mut window)?;
            let mut wnd = BasicWindow::new(glfw, window, renderer, &app_config)?;

            wnd.main_loop(&events);

            Some(())
        })
        .expect("Failed ...");
}

/// Symmetric perspective projection with reverse depth (1.0 -> 0.0) and
/// Vulkan coordinate space.
pub fn perspective(vertical_fov: f32, aspect_ratio: f32, n: f32, f: f32) -> glm::Mat4 {
    let fov_rad = vertical_fov * 2.0f32 * std::f32::consts::PI / 360.0f32;
    let focal_length = 1.0f32 / (fov_rad / 2.0f32).tan();

    let x = focal_length / aspect_ratio;
    let y = -focal_length;
    let a: f32 = n / (f - n);
    let b: f32 = f * a;

    // clang-format off
    glm::Mat4::from_column_slice(&[
        x, 0.0f32, 0.0f32, 0.0f32, 0.0f32, y, 0.0f32, 0.0f32, 0.0f32, 0.0f32, a, -1.0f32, 0.0f32,
        0.0f32, b, 0.0f32,
    ])

    //   if (inverse)
    //   {
    //       *inverse = glm::mat4{
    //           1/x,  0.0f, 0.0f,  0.0f,
    //           0.0f,  1/y, 0.0f,  0.0f,
    //           0.0f, 0.0f, 0.0f, -1.0f,
    //           0.0f, 0.0f,  1/B,   A/B,
    //       };
    //   }
    //
    // // clang-format on
    // return projection;
}
