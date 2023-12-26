use ash::vk::{
    BlendFactor, BlendOp, BufferUsageFlags, ColorComponentFlags, DynamicState, MemoryPropertyFlags,
    PipelineBindPoint, PipelineColorBlendAttachmentState, PrimitiveTopology, ShaderStageFlags,
};

use nalgebra_glm as glm;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use rand::Rng;
use rand_distr::Distribution;

use crate::{
    bindless::BindlessResourceHandle,
    draw_context::{DrawContext, InitContext, UpdateContext},
    vk_renderer::{
        BindlessPipeline, GraphicsPipelineBuilder, ShaderModuleDescription, ShaderModuleSource,
        UniqueBuffer,
    },
    ProgramError,
};

#[derive(serde::Deserialize, serde::Serialize)]
struct ParticlesConfig {
    max_count: u32,
    min_life: std::time::Duration,
    max_life: std::time::Duration,
    min_length: f32,
    max_length: f32,
    min_speed: f32,
    max_speed: f32,
    start_color: [f32; 3],
    end_color: [f32; 3],
    rand_directions: Vec<glm::Vec3>,
}

struct SparksGenerator {
    cfg: ParticlesConfig,
    distribution: rand::distributions::uniform::Uniform<usize>,
}

impl SparksGenerator {
    fn new(init_ctx: &mut InitContext) -> Result<Self, ProgramError> {
        let cfg: ParticlesConfig = ron::de::from_reader(std::fs::File::open(
            init_ctx.cfg.engine.config_path("particles.cfg.ron"),
        )?)?;

        let distribution = rand::distributions::Uniform::<usize>::new(0, cfg.rand_directions.len());

        Ok(Self { cfg, distribution })
    }

    fn generate(
        &mut self,
        position: glm::Vec3,
        direction: glm::Vec3,
    ) -> impl std::iter::Iterator<Item = ImpactSpark> + '_ {
        let mut rng = rand::thread_rng();
        let align_matrix =
            nalgebra::geometry::Rotation3::rotation_between(&direction, &glm::Vec3::z_axis())
                .unwrap_or_else(|| nalgebra::geometry::Rotation3::identity());

        (0..self.cfg.max_count)
            .map(move |_| {
                let eject_dir = glm::normalize(
                    &(align_matrix.transform_vector(
                        &self.cfg.rand_directions[self.distribution.sample(&mut rng)],
                    )),
                );
                let life = rng.gen_range(self.cfg.min_life..=self.cfg.max_life);
                let length = rng.gen_range(self.cfg.min_length..=self.cfg.max_length);
                let speed = rng.gen_range(self.cfg.min_speed..=self.cfg.max_speed);

                ImpactSpark {
                    dir: eject_dir,
                    life,
                    pos: position,
                    speed,
                    length,
                }
            })
            .into_iter()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ImpactSpark {
    pub pos: glm::Vec3,
    pub dir: glm::Vec3,
    life: std::time::Duration,
    speed: f32,
    length: f32,
}

#[derive(Copy, Clone)]
#[repr(C, align(16))]
struct SparkInstance {
    pos: glm::Vec3,
    pad: u32,
    color: glm::Vec3,
    intensity: f32,
}

pub struct SparksSystem {
    instances_buf: UniqueBuffer,
    ssbo_handle: Vec<BindlessResourceHandle>,
    pipeline: BindlessPipeline,
    sparks_cpu: Vec<ImpactSpark>,
    generator: SparksGenerator,
    _file_watcher: RecommendedWatcher,
    rx: std::sync::mpsc::Receiver<std::path::PathBuf>,
}

/// TODO: number of spawned particles should be based on the distance between the shooter and the
/// impact point. The further away the camera is, the fewer particles get spawned
/// TODO: move these to GPU in the future maybe ??!!
impl SparksSystem {
    const MAX_SPARKS: usize = 1024;

    pub fn create(init_ctx: &mut InitContext) -> Result<SparksSystem, ProgramError> {
        let instances_buf = UniqueBuffer::with_capacity(
            init_ctx.renderer,
            BufferUsageFlags::STORAGE_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            Self::MAX_SPARKS,
            std::mem::size_of::<SparkInstance>(),
            init_ctx.renderer.max_inflight_frames(),
        )?;

        init_ctx
            .renderer
            .debug_set_object_tag("ParticleSys/sparks SSBO", &instances_buf);

        let ssbo_handle = init_ctx.rsys.bindless.register_chunked_ssbo(
            init_ctx.renderer,
            &instances_buf,
            init_ctx.renderer.max_inflight_frames() as usize,
        );

        let pipeline = GraphicsPipelineBuilder::new()
            .set_input_assembly_state(PrimitiveTopology::LINE_LIST, false)
            .shader_stages(&[
                ShaderModuleDescription {
                    source: ShaderModuleSource::File(
                        &init_ctx.cfg.engine.shader_path("sparks.vert.spv"),
                    ),
                    stage: ShaderStageFlags::VERTEX,
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(
                        &init_ctx.cfg.engine.shader_path("sparks.frag.spv"),
                    ),
                    entry_point: "main",
                },
            ])
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR])
            .set_colorblend_attachment(
                0,
                PipelineColorBlendAttachmentState::builder()
                    .blend_enable(true)
                    .color_blend_op(BlendOp::ADD)
                    .alpha_blend_op(BlendOp::ADD)
                    .src_color_blend_factor(BlendFactor::ONE)
                    .dst_color_blend_factor(BlendFactor::ONE)
                    .src_alpha_blend_factor(BlendFactor::ONE)
                    .dst_alpha_blend_factor(BlendFactor::ONE)
                    .color_write_mask(ColorComponentFlags::RGBA)
                    .build(),
            )
            .build_bindless(
                init_ctx.renderer.graphics_device(),
                init_ctx.renderer.pipeline_cache(),
                init_ctx.rsys.bindless_setup().pipeline_layout,
                init_ctx.renderer.renderpass(),
                0,
            )?;

        let (tx, rx) = std::sync::mpsc::channel::<std::path::PathBuf>();
        let mut file_watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                if let Ok(e) = event {
                    use notify::event::EventKind;
                    match e.kind {
                        EventKind::Modify(notify::event::ModifyKind::Data(_)) => {
                            if e.paths
                                .iter()
                                .filter_map(|p| p.file_name())
                                .any(|p| p == "particles.cfg.ron")
                            {
                                _ = tx.send(e.paths[0].clone());
                            }
                        }
                        _ => {}
                    }
                }
            })?;

        let watch = file_watcher.watch(&init_ctx.cfg.engine.config, RecursiveMode::Recursive);
        let _ = watch;

        Ok(SparksSystem {
            instances_buf,
            ssbo_handle,
            pipeline,
            sparks_cpu: Vec::new(),
            generator: SparksGenerator::new(init_ctx)?,
            rx,
            _file_watcher: file_watcher,
        })
    }

    pub fn spawn_sparks(&mut self, position: glm::Vec3, direction: glm::Vec3) {
        self.sparks_cpu
            .extend(self.generator.generate(position, direction));
    }

    pub fn update(&mut self, update_context: &mut UpdateContext) {
        while let Ok(e) = self.rx.try_recv() {
            log::info!("Reloading particle configuration file: {}", e.display());

            let _ = std::fs::File::open(update_context.cfg.engine.config_path("particles.cfg.ron"))
                .and_then(|cfg_file| {
                    if let Ok(cfg) =
                        ron::de::from_reader::<std::fs::File, ParticlesConfig>(cfg_file)
                    {
                        self.generator.cfg = cfg;
                    }
                    Ok(())
                });
        }

        self.sparks_cpu.retain_mut(|s| {
            if s.life > update_context.elapsed_time {
                s.life -= update_context.elapsed_time;
                s.pos += s.dir * s.speed * update_context.frame_time as f32;
                true
            } else {
                false
            }
        });
    }

    pub fn draw(&self, draw_context: &DrawContext) {
        if self.sparks_cpu.len() == 0 {
            return;
        }

        let _ = self
            .instances_buf
            .map_for_frame(draw_context.renderer, draw_context.frame_id)
            .map(|instances| unsafe {
                let gpu = instances.memptr() as *mut SparkInstance;

                let _ = self.sparks_cpu.iter().fold(0isize, |offset, s| {
                    let start_p = SparkInstance {
                        pos: s.pos,
                        color: [1f32, 1f32, 1f32].into(),
                        pad: 0xdeadbeef,
                        intensity: 1f32,
                    };

                    gpu.offset(offset).write(start_p);

                    let end_p = SparkInstance {
                        pos: s.pos + s.dir * s.length,
                        color: [1f32, 0.84f32, 0f32].into(),
                        pad: 0xdeadbeef,
                        intensity: 1f32,
                    };

                    gpu.offset(offset + 1).write(end_p);

                    offset + 2
                });
            });

        unsafe {
            draw_context.renderer.graphics_device().cmd_bind_pipeline(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.handle,
            );
            draw_context.renderer.graphics_device().cmd_set_viewport(
                draw_context.cmd_buff,
                0,
                &[draw_context.viewport],
            );
            draw_context.renderer.graphics_device().cmd_set_scissor(
                draw_context.cmd_buff,
                0,
                &[draw_context.scissor],
            );

            let push_const = draw_context.global_ubo_handle << 16
                | self.ssbo_handle[draw_context.frame_id as usize].handle();

            draw_context.renderer.graphics_device().cmd_push_constants(
                draw_context.cmd_buff,
                draw_context.rsys.bindless_setup().pipeline_layout,
                ShaderStageFlags::ALL,
                0,
                &push_const.to_le_bytes(),
            );

            draw_context.renderer.graphics_device().cmd_draw(
                draw_context.cmd_buff,
                2,
                self.sparks_cpu.len() as u32,
                0,
                0,
            );
        }
    }
}
