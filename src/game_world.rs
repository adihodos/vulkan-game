use std::{cell::RefCell, mem::size_of};

use ash::vk::{
    BorderColor, BufferUsageFlags, DescriptorBufferInfo, DescriptorImageInfo, DescriptorSet,
    DescriptorSetAllocateInfo, DescriptorType, DeviceSize, Filter, ImageLayout, IndexType,
    MemoryPropertyFlags, PipelineBindPoint, SamplerAddressMode, SamplerCreateInfo,
    SamplerMipmapMode, WriteDescriptorSet,
};
use glm::Mat4;
use nalgebra_glm::Vec4;

use nalgebra_glm as glm;
use smallvec::SmallVec;

use crate::{
    app_config::AppConfig,
    draw_context::DrawContext,
    resource_cache::{PbrDescriptorType, PbrRenderableHandle, ResourceHolder},
    skybox::Skybox,
    vk_renderer::{ScopedBufferMapping, UniqueBuffer, UniqueSampler, VulkanRenderer},
};

#[derive(Copy, Clone, Debug)]
struct DrawOpts {
    wireframe_color: Vec4,
    draw_normals: bool,
    normals_color: Vec4,
}

#[repr(C)]
pub struct PbrLightingData {
    pub eye_pos: glm::Vec3,
}

#[repr(C)]
pub struct PbrTransformDataUBO {
    pub projection: glm::Mat4,
    pub view: glm::Mat4,
    pub world: glm::Mat4,
}

struct PbrCpu2GpuData {
    aligned_ubo_transforms_size: DeviceSize,
    aligned_ubo_lighting_size: DeviceSize,
    size_ubo_transforms_one_frame: DeviceSize,
    size_ubo_lighting_one_frame: DeviceSize,
    ubo_transforms: UniqueBuffer,
    ubo_lighting: UniqueBuffer,
    object_descriptor_sets: Vec<DescriptorSet>,
    ibl_descriptor_sets: Vec<DescriptorSet>,
    samplers: Vec<UniqueSampler>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
struct GameObjectHandle(u32);

struct GameObjectData {
    handle: GameObjectHandle,
    renderable: PbrRenderableHandle,
}

pub struct GameWorld {
    draw_opts: RefCell<DrawOpts>,
    resource_cache: ResourceHolder,
    skybox: Skybox,
    pbr_cpu_2_gpu: PbrCpu2GpuData,
    objects: Vec<GameObjectData>,
}

impl GameWorld {
    pub fn new(renderer: &VulkanRenderer, app_cfg: &AppConfig) -> Option<GameWorld> {
        let skybox = Skybox::create(renderer, &app_cfg.scene, &app_cfg.engine)?;

        let resource_cache = ResourceHolder::create(renderer, app_cfg)?;
        let aligned_ubo_transforms_size = VulkanRenderer::aligned_size_of_type::<PbrTransformDataUBO>(
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
        );
        let size_ubo_transforms_one_frame = aligned_ubo_transforms_size * 1; // 1 object for now
        let pbr_ubo_transforms = UniqueBuffer::new(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            size_ubo_transforms_one_frame * renderer.max_inflight_frames() as DeviceSize,
        )?;

        let aligned_ubo_lighting_size = VulkanRenderer::aligned_size_of_type::<PbrLightingData>(
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
        );
        let size_ubo_lighting_one_frame = aligned_ubo_lighting_size * 1; // 1 object for now
        let pbr_ubo_lighting = UniqueBuffer::new(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            size_ubo_lighting_one_frame * renderer.max_inflight_frames() as DeviceSize,
        )?;

        let pbr_descriptor_layouts = resource_cache.pbr_pipeline().descriptor_layouts();

        let per_object_ds_layouts = [
            pbr_descriptor_layouts[PbrDescriptorType::VsTransformsUbo as usize],
            pbr_descriptor_layouts[PbrDescriptorType::FsLightingData as usize],
        ];

        log::info!("PBR desc layouts {:?}", pbr_descriptor_layouts);
        log::info!("Per object layouts: {:?}", per_object_ds_layouts);

        let object_pbr_descriptor_sets = unsafe {
            renderer.graphics_device().allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .set_layouts(&per_object_ds_layouts)
                    .descriptor_pool(renderer.descriptor_pool())
                    .build(),
            )
        }
        .expect("Papali, papali sukyyyyyyyyyyyyyy");

        log::info!(
            "Descriptor sets transforms + UBO : {:?}",
            object_pbr_descriptor_sets
        );

        let desc_buff_info = [
            DescriptorBufferInfo::builder()
                .buffer(pbr_ubo_transforms.buffer)
                .offset(0)
                .range(size_of::<PbrTransformDataUBO>() as DeviceSize)
                .build(),
            DescriptorBufferInfo::builder()
                .buffer(pbr_ubo_lighting.buffer)
                .offset(0)
                .range(size_of::<PbrLightingData>() as DeviceSize)
                .build(),
        ];

        log::info!("DS updates: {:?}", desc_buff_info);

        let write_descriptors_transforms_lighting = [
            WriteDescriptorSet::builder()
                .dst_set(object_pbr_descriptor_sets[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&desc_buff_info[0..1])
                .build(),
            WriteDescriptorSet::builder()
                .dst_set(object_pbr_descriptor_sets[1])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&desc_buff_info[1..])
                .build(),
        ];

        unsafe {
            renderer
                .graphics_device()
                .update_descriptor_sets(&write_descriptors_transforms_lighting, &[]);
        }

        let mut samplers_ibl = SmallVec::<[UniqueSampler; 8]>::new();
        let mut ibl_descriptor_sets = SmallVec::<[DescriptorSet; 4]>::new();

        let sampler_brdf_lut = UniqueSampler::new(
            renderer.graphics_device(),
            &SamplerCreateInfo::builder()
                .min_lod(0f32)
                .max_lod(1f32)
                .min_filter(Filter::LINEAR)
                .mag_filter(Filter::LINEAR)
                .mipmap_mode(SamplerMipmapMode::LINEAR)
                .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                .border_color(BorderColor::INT_OPAQUE_BLACK)
                .max_anisotropy(1f32)
                .build(),
        )?;

        skybox.get_ibl_data().iter().for_each(|ibl_data| {
            let levels_irradiance = ibl_data.irradiance.0.info.num_levels;

            let sampler_cubemaps = UniqueSampler::new(
                renderer.graphics_device(),
                &SamplerCreateInfo::builder()
                    .min_lod(0f32)
                    .max_lod(levels_irradiance as f32)
                    .min_filter(Filter::LINEAR)
                    .mag_filter(Filter::LINEAR)
                    .mipmap_mode(SamplerMipmapMode::LINEAR)
                    .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                    .border_color(BorderColor::INT_OPAQUE_BLACK)
                    .max_anisotropy(1f32)
                    .build(),
            )
            .expect("Failed to create sampler");

            let ibl_desc_img_info = [
                DescriptorImageInfo::builder()
                    .image_view(ibl_data.irradiance.1.view)
                    .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .sampler(sampler_cubemaps.sampler)
                    .build(),
                DescriptorImageInfo::builder()
                    .image_view(ibl_data.specular.1.view)
                    .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .sampler(sampler_cubemaps.sampler)
                    .build(),
                DescriptorImageInfo::builder()
                    .image_view(ibl_data.brdf_lut.1.view)
                    .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .sampler(sampler_brdf_lut.sampler)
                    .build(),
            ];

            samplers_ibl.push(sampler_cubemaps);

            let dset_ibl = unsafe {
                renderer.graphics_device().allocate_descriptor_sets(
                    &DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(renderer.descriptor_pool())
                        .set_layouts(&pbr_descriptor_layouts[3..])
                        .build(),
                )
            }
            .expect("Papalyyyy cykyyyyyyyyyyyyyyy");

            let wds = [
                //
                // irradiance
                WriteDescriptorSet::builder()
                    .dst_set(dset_ibl[0])
                    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_binding(0)
                    .image_info(&ibl_desc_img_info[0..1])
                    .dst_array_element(0)
                    .build(),
                //
                // specular
                WriteDescriptorSet::builder()
                    .dst_set(dset_ibl[0])
                    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .image_info(&ibl_desc_img_info[1..2])
                    .build(),
                //
                // BRDF lut
                WriteDescriptorSet::builder()
                    .dst_set(dset_ibl[0])
                    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .image_info(&ibl_desc_img_info[2..])
                    .build(),
            ];

            unsafe {
                renderer.graphics_device().update_descriptor_sets(&wds, &[]);
            }

            ibl_descriptor_sets.extend(dset_ibl);
        });

        samplers_ibl.push(sampler_brdf_lut);
        let objects = vec![GameObjectData {
            handle: GameObjectHandle(0),
            renderable: resource_cache.get_geometry_handle(&"sa23"),
        }];

        Some(GameWorld {
            draw_opts: RefCell::new(DrawOpts {
                wireframe_color: Vec4::new(0f32, 1f32, 0f32, 1f32),
                draw_normals: false,
                normals_color: Vec4::new(1f32, 0f32, 0f32, 1f32),
            }),
            resource_cache,
            skybox,
            pbr_cpu_2_gpu: PbrCpu2GpuData {
                aligned_ubo_transforms_size,
                aligned_ubo_lighting_size,
                size_ubo_transforms_one_frame,
                size_ubo_lighting_one_frame,
                ubo_transforms: pbr_ubo_transforms,
                ubo_lighting: pbr_ubo_lighting,
                object_descriptor_sets: object_pbr_descriptor_sets,
                ibl_descriptor_sets: ibl_descriptor_sets.into_vec(),
                samplers: samplers_ibl.into_vec(),
            },
            objects,
        })
    }

    pub fn draw(&self, draw_context: &DrawContext) {
        self.skybox.draw(draw_context);

        let device = draw_context.renderer.graphics_device();

        let viewports = [draw_context.viewport];
        let scisssors = [draw_context.scissor];

        let view_matrix = draw_context.camera.view_transform();

        let perspective = draw_context.projection;

        let transforms = PbrTransformDataUBO {
            world: Mat4::identity(),
            view: draw_context.camera.view_transform(),
            projection: perspective,
        };

        ScopedBufferMapping::create(
            draw_context.renderer,
            &self.pbr_cpu_2_gpu.ubo_transforms,
            self.pbr_cpu_2_gpu.size_ubo_transforms_one_frame,
            self.pbr_cpu_2_gpu.size_ubo_transforms_one_frame * draw_context.frame_id as u64,
        )
        .map(|mapping| unsafe {
            std::ptr::copy_nonoverlapping(
                &transforms as *const _,
                mapping.memptr() as *mut PbrTransformDataUBO,
                1,
            );
        });

        let pbr_light_data = PbrLightingData {
            eye_pos: draw_context.camera.position(),
        };

        ScopedBufferMapping::create(
            draw_context.renderer,
            &self.pbr_cpu_2_gpu.ubo_lighting,
            self.pbr_cpu_2_gpu.size_ubo_lighting_one_frame,
            self.pbr_cpu_2_gpu.size_ubo_lighting_one_frame * draw_context.frame_id as u64,
        )
        .map(|mapping| unsafe {
            std::ptr::copy_nonoverlapping(
                &pbr_light_data as *const _,
                mapping.memptr() as *mut PbrLightingData,
                1,
            );
        });

        unsafe {
            device.cmd_bind_pipeline(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.resource_cache.pbr_pipeline().pipeline,
            );
            device.cmd_set_viewport(draw_context.cmd_buff, 0, &viewports);
            device.cmd_set_scissor(draw_context.cmd_buff, 0, &scisssors);

            let sa23_renderable = self
                .resource_cache
                .get_pbr_renderable(self.objects[0].renderable);

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

            let bound_descriptor_sets = [
                self.pbr_cpu_2_gpu.object_descriptor_sets[0],
                sa23_renderable.descriptor_sets[0],
                self.pbr_cpu_2_gpu.object_descriptor_sets[1],
                self.pbr_cpu_2_gpu.ibl_descriptor_sets[self.skybox.active_skybox as usize],
            ];

            let descriptor_set_offsets = [
                self.pbr_cpu_2_gpu.size_ubo_transforms_one_frame as u32 * draw_context.frame_id,
                0,
                self.pbr_cpu_2_gpu.size_ubo_lighting_one_frame as u32 * draw_context.frame_id,
            ];

            device.cmd_bind_descriptor_sets(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.resource_cache.pbr_pipeline().layout,
                0,
                &bound_descriptor_sets,
                &descriptor_set_offsets,
            );

            device.cmd_draw_indexed(
                draw_context.cmd_buff,
                sa23_renderable.geometry.index_count,
                1,
                sa23_renderable.geometry.index_offset,
                sa23_renderable.geometry.vertex_offset as i32,
                0,
            );
        }
    }

    pub fn ui(&self, ui: &mut imgui::Ui) {
        let choices = ["test test this is 1", "test test this is 2"];
        ui.window("Hello world")
            .size([300.0, 110.0], imgui::Condition::FirstUseEver)
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
