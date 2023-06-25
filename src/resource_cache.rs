use ash::vk::{
    BufferUsageFlags, CullModeFlags, DescriptorBufferInfo, DescriptorImageInfo, DescriptorSet,
    DescriptorSetAllocateInfo, DescriptorSetLayoutBinding, DescriptorType, DeviceSize,
    DynamicState, Filter, Format, FrontFace, ImageLayout, MemoryPropertyFlags,
    PipelineRasterizationStateCreateInfo, PolygonMode, SamplerAddressMode, SamplerCreateInfo,
    SamplerMipmapMode, ShaderStageFlags, VertexInputAttributeDescription,
    VertexInputBindingDescription, VertexInputRate, WriteDescriptorSet,
};
use chrono::Duration;
use memoffset::offset_of;

use rayon::prelude::*;
use smallvec::SmallVec;
use std::{collections::HashMap, mem::size_of, path::Path, time::Instant};

use crate::{
    app_config::AppConfig,
    imported_geometry::{GeometryNode, GeometryVertex, ImportedGeometry},
    math::AABB3,
    pbr::{PbrMaterial, PbrMaterialTextureCollection},
    vk_renderer::{
        GraphicsPipelineBuilder, GraphicsPipelineLayoutBuilder, ShaderModuleDescription,
        ShaderModuleSource, UniqueBuffer, UniqueGraphicsPipeline, UniqueSampler, VulkanRenderer,
    },
};

#[derive(Clone, Debug, Default)]
pub struct GeometryRenderInfo {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub pbr_data_offset: u32,
    pub pbr_data_range: u32,
    pub nodes: Vec<GeometryNode>,
    pub aabb: AABB3,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
pub struct PbrRenderableHandle(u32);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum PbrDescriptorType {
    VsTransformsUbo,
    //
    // per-geometry materials
    FsPbrObjectData,
    FsLightingData,
    //
    // Skybox IBL images
    FsSkyboxIblData,
}

pub struct PbrRenderable {
    pub geometry: GeometryRenderInfo,
    materials: PbrMaterialTextureCollection,
    pub descriptor_sets: Vec<DescriptorSet>,
}

pub struct ResourceHolder {
    vertex_buffer: UniqueBuffer,
    index_buffer: UniqueBuffer,
    pbr_data_buffer: UniqueBuffer,
    sampler: UniqueSampler,
    pipeline: UniqueGraphicsPipeline,
    pipeline_instanced: UniqueGraphicsPipeline,
    handles: HashMap<String, PbrRenderableHandle>,
    geometries: Vec<PbrRenderable>,
}

impl ResourceHolder {
    pub fn pbr_pipeline(&self) -> &UniqueGraphicsPipeline {
        &self.pipeline
    }

    pub fn get_geometry_handle(&self, name: &str) -> PbrRenderableHandle {
        *self.handles.get(name).unwrap()
    }

    pub fn get_geometry_info(&self, handle: PbrRenderableHandle) -> &GeometryRenderInfo {
        &self.geometries[handle.0 as usize].geometry
    }

    pub fn get_pbr_renderable(&self, handle: PbrRenderableHandle) -> &PbrRenderable {
        &self.geometries[handle.0 as usize]
    }

    pub fn vertex_buffer(&self) -> ash::vk::Buffer {
        self.vertex_buffer.buffer
    }

    pub fn index_buffer(&self) -> ash::vk::Buffer {
        self.index_buffer.buffer
    }

    pub fn create(renderer: &VulkanRenderer, app_config: &AppConfig) -> Option<ResourceHolder> {
        let s = Instant::now();

        let imported_geometries = app_config
            .scene
            .geometry
            .par_iter()
            .map(|gdata| {
                let geometry_path = Path::new(&app_config.engine.models).join(&gdata.path);
                let imported_geom = ImportedGeometry::import_from_file(&geometry_path).expect(
                    &format!("Failed to load model: {}", geometry_path.to_str().unwrap()),
                );

                (&gdata.tag, imported_geom)
            })
            .collect::<Vec<_>>();

        let e = Duration::from_std(s.elapsed()).unwrap();
        log::info!(
            "Loaded {} geometries in {}m {}s {}ms",
            imported_geometries.len(),
            e.num_minutes(),
            e.num_seconds(),
            e.num_milliseconds()
        );

        let mut handles = HashMap::<String, PbrRenderableHandle>::new();
        let mut pbr_data = Vec::<PbrMaterial>::new();
        let mut pbr_textures = Vec::<PbrMaterialTextureCollection>::new();
        let mut geometry = Vec::<GeometryRenderInfo>::new();
        let (mut vertex_offset, mut index_offset) = (0u32, 0u32);
        let pbr_data_aligned_size = VulkanRenderer::aligned_size_of_type::<PbrMaterial>(
            renderer.device_properties().limits.non_coherent_atom_size,
        );

        imported_geometries.iter().for_each(|(tag, geom)| {
            log::info!("{} -> {}", tag, geom.aabb.extents());
            let texture_cpu2gpu_copy_work_package = renderer
                .create_work_package()
                .expect("Failed to create work package");

            let pbr_mtl_tex = PbrMaterialTextureCollection::create(
                renderer,
                geom.pbr_base_color_images(),
                geom.pbr_metallic_roughness_images(),
                geom.pbr_normal_images(),
                &texture_cpu2gpu_copy_work_package,
            )
            .expect("Failed to create pbr materials");

            renderer.push_work_package(texture_cpu2gpu_copy_work_package);

            let geometry_handle = PbrRenderableHandle(geometry.len() as u32);
            let pbr_data_offset =
                (size_of::<PbrMaterial>() as DeviceSize * (pbr_data.len() as DeviceSize)) as u32;

            pbr_textures.push(pbr_mtl_tex);
            geometry.push(GeometryRenderInfo {
                vertex_offset,
                index_offset,
                index_count: geom.index_count(),
                pbr_data_offset,
                pbr_data_range: (geom.pbr_materials().len() * size_of::<PbrMaterial>()) as u32,
                nodes: geom.nodes().to_vec(),
                aabb: geom.aabb,
            });
            pbr_data.extend(geom.pbr_materials().iter());

            vertex_offset += geom.vertex_count();
            index_offset += geom.index_count();

            handles.insert((*tag).clone(), geometry_handle);
        });

        let _vertex_bytes = vertex_offset as DeviceSize * size_of::<GeometryVertex>() as DeviceSize;
        let vertex_data: SmallVec<[&[GeometryVertex]; 8]> = imported_geometries
            .iter()
            .map(|(_, geom)| geom.vertices())
            .collect();

        let vertex_buffer = UniqueBuffer::gpu_only_buffer(
            renderer,
            BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            MemoryPropertyFlags::DEVICE_LOCAL,
            &vertex_data,
            None,
        )?;

        let _index_bytes = index_offset as DeviceSize * size_of::<u32>() as DeviceSize;
        let indices_data: SmallVec<[&[u32]; 8]> = imported_geometries
            .iter()
            .map(|(_, geom)| geom.indices())
            .collect();

        let index_buffer = UniqueBuffer::gpu_only_buffer(
            renderer,
            BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            MemoryPropertyFlags::DEVICE_LOCAL,
            &indices_data,
            None,
        )?;

        let _pbr_bytes = pbr_data_aligned_size * pbr_data.len() as DeviceSize;
        let pbr_data_buffer = UniqueBuffer::gpu_only_buffer(
            renderer,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            MemoryPropertyFlags::DEVICE_LOCAL,
            &[&pbr_data],
            Some(renderer.device_properties().limits.non_coherent_atom_size),
        )?;

        let pipeline = Self::create_rendering_pipeline(renderer, app_config)?;

        let sampler = UniqueSampler::new(
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
                .border_color(ash::vk::BorderColor::INT_OPAQUE_BLACK)
                .max_anisotropy(1f32)
                .build(),
        )?;

        assert_eq!(geometry.len(), pbr_textures.len());
        let geometries = geometry
            .into_iter()
            .zip(pbr_textures.into_iter())
            .map(|(rend_geom, pbr_mtl)| {
                let descriptor_sets = unsafe {
                    renderer.graphics_device().allocate_descriptor_sets(
                        &DescriptorSetAllocateInfo::builder()
                            .descriptor_pool(renderer.descriptor_pool())
                            .set_layouts(
                                &pipeline.descriptor_layouts()[PbrDescriptorType::FsPbrObjectData
                                    as usize
                                    ..PbrDescriptorType::FsLightingData as usize],
                            )
                            .build(),
                    )
                }
                .expect(&format!("Failed to allocate descriptor sets: {}", e));

                let buff_info = [DescriptorBufferInfo::builder()
                    .buffer(pbr_data_buffer.buffer)
                    .offset(rend_geom.pbr_data_offset as DeviceSize)
                    .range(rend_geom.pbr_data_range as DeviceSize)
                    .build()];

                let image_info = [
                    DescriptorImageInfo::builder()
                        .image_view(pbr_mtl.base_color_imageview.view)
                        .sampler(sampler.sampler)
                        .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .build(),
                    DescriptorImageInfo::builder()
                        .image_view(pbr_mtl.metallic_imageview.view)
                        .sampler(sampler.sampler)
                        .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .build(),
                    DescriptorImageInfo::builder()
                        .image_view(pbr_mtl.normal_imageview.view)
                        .sampler(sampler.sampler)
                        .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .build(),
                ];

                let target_desc_set = descriptor_sets[0];
                let wds = [
                    WriteDescriptorSet::builder()
                        .dst_set(target_desc_set)
                        .dst_binding(0)
                        .buffer_info(&buff_info)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                        .build(),
                    WriteDescriptorSet::builder()
                        .dst_set(target_desc_set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&image_info[0..1])
                        .build(),
                    WriteDescriptorSet::builder()
                        .dst_set(target_desc_set)
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&image_info[1..2])
                        .build(),
                    WriteDescriptorSet::builder()
                        .dst_set(target_desc_set)
                        .dst_binding(3)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&image_info[2..])
                        .build(),
                ];

                unsafe {
                    renderer.graphics_device().update_descriptor_sets(&wds, &[]);
                }

                PbrRenderable {
                    geometry: rend_geom,
                    materials: pbr_mtl,
                    descriptor_sets,
                }
            })
            .collect::<Vec<_>>();

        Some(ResourceHolder {
            vertex_buffer,
            index_buffer,
            pbr_data_buffer,
            sampler,
            pipeline,
            pipeline_instanced: Self::create_instanced_rendering_pipeline(renderer, app_config)?,
            handles,
            geometries,
        })
    }

    fn create_rendering_pipeline(
        renderer: &VulkanRenderer,
        app_config: &AppConfig,
    ) -> Option<UniqueGraphicsPipeline> {
        GraphicsPipelineBuilder::new()
            .add_vertex_input_attribute_descriptions(&[
                VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: Format::R32G32B32_SFLOAT,
                    offset: offset_of!(GeometryVertex, pos) as u32,
                },
                VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: Format::R32G32B32_SFLOAT,
                    offset: offset_of!(GeometryVertex, normal) as u32,
                },
                VertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: Format::R32G32_SFLOAT,
                    offset: offset_of!(GeometryVertex, uv) as u32,
                },
                VertexInputAttributeDescription {
                    location: 3,
                    binding: 0,
                    format: Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(GeometryVertex, color) as u32,
                },
                VertexInputAttributeDescription {
                    location: 4,
                    binding: 0,
                    format: Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(GeometryVertex, tangent) as u32,
                },
                VertexInputAttributeDescription {
                    location: 5,
                    binding: 0,
                    format: Format::R32_UINT,
                    offset: offset_of!(GeometryVertex, pbr_buf_id) as u32,
                },
            ])
            .add_vertex_input_attribute_binding(
                VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(size_of::<GeometryVertex>() as u32)
                    .input_rate(VertexInputRate::VERTEX)
                    .build(),
            )
            .shader_stages(&[
                ShaderModuleDescription {
                    stage: ShaderStageFlags::VERTEX,
                    source: ShaderModuleSource::File(
                        &app_config.engine.shader_path("pbr.vert.spv"),
                    ),
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(
                        &app_config.engine.shader_path("pbr.frag.spv"),
                    ),
                    entry_point: "main",
                },
            ])
            .set_raster_state(
                PipelineRasterizationStateCreateInfo::builder()
                    .cull_mode(CullModeFlags::BACK)
                    .front_face(FrontFace::CLOCKWISE)
                    .line_width(1f32)
                    .polygon_mode(PolygonMode::FILL)
                    .build(),
            )
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR])
            .build(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                GraphicsPipelineLayoutBuilder::new()
                    //
                    // set 0
                    .set(
                        0,
                        &[DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .descriptor_count(1)
                            .stage_flags(ShaderStageFlags::VERTEX)
                            .build()],
                    )
                    //
                    //set 1
                    .set(
                        1,
                        &[
                            DescriptorSetLayoutBinding::builder()
                                .binding(0)
                                .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                            DescriptorSetLayoutBinding::builder()
                                .binding(1)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                            DescriptorSetLayoutBinding::builder()
                                .binding(2)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                            DescriptorSetLayoutBinding::builder()
                                .binding(3)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                        ],
                    )
                    //
                    //set 2
                    .set(
                        2,
                        &[DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .descriptor_count(1)
                            .stage_flags(ShaderStageFlags::FRAGMENT)
                            .build()],
                    )
                    //
                    //set 3
                    .set(
                        3,
                        &[
                            DescriptorSetLayoutBinding::builder()
                                .binding(0)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                            DescriptorSetLayoutBinding::builder()
                                .binding(1)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                            DescriptorSetLayoutBinding::builder()
                                .binding(2)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                        ],
                    )
                    .build(renderer.graphics_device())?,
                renderer.renderpass(),
                0,
            )
    }

    fn create_instanced_rendering_pipeline(
        renderer: &VulkanRenderer,
        app_config: &AppConfig,
    ) -> Option<UniqueGraphicsPipeline> {
        GraphicsPipelineBuilder::new()
            .add_vertex_input_attribute_descriptions(&[
                VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: Format::R32G32B32_SFLOAT,
                    offset: offset_of!(GeometryVertex, pos) as u32,
                },
                VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: Format::R32G32B32_SFLOAT,
                    offset: offset_of!(GeometryVertex, normal) as u32,
                },
                VertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: Format::R32G32_SFLOAT,
                    offset: offset_of!(GeometryVertex, uv) as u32,
                },
                VertexInputAttributeDescription {
                    location: 3,
                    binding: 0,
                    format: Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(GeometryVertex, color) as u32,
                },
                VertexInputAttributeDescription {
                    location: 4,
                    binding: 0,
                    format: Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(GeometryVertex, tangent) as u32,
                },
                VertexInputAttributeDescription {
                    location: 5,
                    binding: 0,
                    format: Format::R32_UINT,
                    offset: offset_of!(GeometryVertex, pbr_buf_id) as u32,
                },
            ])
            .add_vertex_input_attribute_binding(
                VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(size_of::<GeometryVertex>() as u32)
                    .input_rate(VertexInputRate::VERTEX)
                    .build(),
            )
            .shader_stages(&[
                ShaderModuleDescription {
                    stage: ShaderStageFlags::VERTEX,
                    source: ShaderModuleSource::File(
                        &app_config.engine.shader_path("pbr.instanced.vert.spv"),
                    ),
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(
                        &app_config.engine.shader_path("pbr.frag.spv"),
                    ),
                    entry_point: "main",
                },
            ])
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR])
            .set_raster_state(
                PipelineRasterizationStateCreateInfo::builder()
                    .cull_mode(CullModeFlags::BACK)
                    .front_face(FrontFace::CLOCKWISE)
                    .line_width(1f32)
                    .polygon_mode(PolygonMode::FILL)
                    .build(),
            )
            .build(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                GraphicsPipelineLayoutBuilder::new()
                    //
                    // set 0
                    .set(
                        0,
                        &[
                            DescriptorSetLayoutBinding::builder()
                                .binding(0)
                                .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::VERTEX)
                                .build(),
                            DescriptorSetLayoutBinding::builder()
                                .binding(1)
                                .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::VERTEX)
                                .build(),
                        ],
                    )
                    //
                    //set 1
                    .set(
                        1,
                        &[
                            DescriptorSetLayoutBinding::builder()
                                .binding(0)
                                .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                            DescriptorSetLayoutBinding::builder()
                                .binding(1)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                            DescriptorSetLayoutBinding::builder()
                                .binding(2)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                            DescriptorSetLayoutBinding::builder()
                                .binding(3)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                        ],
                    )
                    //
                    //set 2
                    .set(
                        2,
                        &[DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .descriptor_count(1)
                            .stage_flags(ShaderStageFlags::FRAGMENT)
                            .build()],
                    )
                    //
                    //set 3
                    .set(
                        3,
                        &[
                            DescriptorSetLayoutBinding::builder()
                                .binding(0)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                            DescriptorSetLayoutBinding::builder()
                                .binding(1)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                            DescriptorSetLayoutBinding::builder()
                                .binding(2)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .build(),
                        ],
                    )
                    .build(renderer.graphics_device())?,
                renderer.renderpass(),
                0,
            )
    }

    pub fn pbr_pipeline_instanced(&self) -> &UniqueGraphicsPipeline {
        &self.pipeline_instanced
    }
}
