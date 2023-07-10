use ash::vk::{
    BufferUsageFlags, CullModeFlags, DescriptorBindingFlags, DescriptorBufferInfo,
    DescriptorImageInfo, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet,
    DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorSetLayoutBindingFlagsCreateInfo, DescriptorSetLayoutCreateInfo, DescriptorType,
    DeviceSize, DrawIndexedIndirectCommand, DynamicState, Extent3D, Filter, Format, FrontFace,
    ImageLayout, ImageTiling, ImageType, ImageUsageFlags, IndexType, MemoryPropertyFlags,
    PipelineBindPoint, PipelineLayout, PipelineLayoutCreateInfo,
    PipelineRasterizationStateCreateInfo, PolygonMode, PushConstantRange, SampleCountFlags,
    SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, ShaderStageFlags, SharingMode,
    VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
    WriteDescriptorSet,
};
use chrono::Duration;
use memoffset::offset_of;

use rayon::prelude::*;
use smallvec::SmallVec;
use std::{collections::HashMap, mem::size_of, path::Path, time::Instant};

use crate::{
    app_config::AppConfig,
    draw_context::{DrawContext, InitContext},
    imported_geometry::{GeometryNode, GeometryVertex, ImportedGeometry},
    math::AABB3,
    pbr::{PbrMaterial, PbrMaterialTextureCollection},
    vk_renderer::{
        Cpu2GpuBuffer, GraphicsPipelineBuilder, GraphicsPipelineLayoutBuilder,
        ShaderModuleDescription, ShaderModuleSource, UniqueBuffer, UniqueGraphicsPipeline,
        UniqueImage, UniqueImageView, UniqueImageWithView, UniqueSampler, VulkanRenderer,
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

impl GeometryRenderInfo {
    pub fn get_node_by_name(&self, id: &str) -> &GeometryNode {
        self.nodes
            .iter()
            .find(|n| n.name == id)
            .expect(&format!("Node {id} not found"))
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
pub struct PbrRenderableHandle(u32);

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
pub struct NonPbrRenderableHandle(u32);

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

pub struct NonPbrRenderable {
    pub geometry: GeometryRenderInfo,
    // pub descriptor_sets: Vec<DescriptorSet>,
}

struct NonPbrResources {
    vertex_buffer: UniqueBuffer,
    index_buffer: UniqueBuffer,
    pipeline_single: UniqueGraphicsPipeline,
    pipeline_instanced: UniqueGraphicsPipeline,
    handles: HashMap<String, NonPbrRenderableHandle>,
    geometries: Vec<NonPbrRenderable>,
}

pub struct ResourceHolder {
    vertex_buffer: UniqueBuffer,
    index_buffer: UniqueBuffer,
    pbr_data_buffer: UniqueBuffer,
    sampler: UniqueSampler,
    pipeline: UniqueGraphicsPipeline,
    pipeline_instanced: UniqueGraphicsPipeline,
    pub handles: HashMap<String, PbrRenderableHandle>,
    geometries: Vec<PbrRenderable>,
    non_pbr: NonPbrResources,
}

impl ResourceHolder {
    pub fn non_pbr_pipeline_single(&self) -> &UniqueGraphicsPipeline {
        &self.non_pbr.pipeline_single
    }

    pub fn non_pbr_pipeline_instanced(&self) -> &UniqueGraphicsPipeline {
        &self.non_pbr.pipeline_instanced
    }

    pub fn pbr_pipeline(&self) -> &UniqueGraphicsPipeline {
        &self.pipeline
    }

    pub fn get_pbr_geometry_handle(&self, name: &str) -> PbrRenderableHandle {
        *self.handles.get(name).unwrap()
    }

    pub fn get_pbr_geometry_info(&self, handle: PbrRenderableHandle) -> &GeometryRenderInfo {
        &self.geometries[handle.0 as usize].geometry
    }

    pub fn get_pbr_renderable(&self, handle: PbrRenderableHandle) -> &PbrRenderable {
        &self.geometries[handle.0 as usize]
    }

    pub fn vertex_buffer_pbr(&self) -> ash::vk::Buffer {
        self.vertex_buffer.buffer
    }

    pub fn index_buffer_pbr(&self) -> ash::vk::Buffer {
        self.index_buffer.buffer
    }

    pub fn get_non_pbr_geometry_handle(&self, name: &str) -> NonPbrRenderableHandle {
        *self
            .non_pbr
            .handles
            .get(name)
            .expect(&format!("Non-pbr geometry {name} not found"))
    }

    pub fn get_non_pbr_geometry_info(&self, handle: NonPbrRenderableHandle) -> &GeometryRenderInfo {
        &self.non_pbr.geometries[handle.0 as usize].geometry
    }

    pub fn get_non_pbr_renderable(&self, handle: NonPbrRenderableHandle) -> &NonPbrRenderable {
        &self.non_pbr.geometries[handle.0 as usize]
    }

    pub fn vertex_buffer_non_pbr(&self) -> ash::vk::Buffer {
        self.non_pbr.vertex_buffer.buffer
    }

    pub fn index_buffer_non_pbr(&self) -> ash::vk::Buffer {
        self.non_pbr.index_buffer.buffer
    }

    fn create_non_pbr_resources(
        renderer: &VulkanRenderer,
        app_config: &AppConfig,
        resources: &[(&String, ImportedGeometry)],
    ) -> Option<NonPbrResources> {
        let mut handles = HashMap::<String, NonPbrRenderableHandle>::new();
        let mut geometry = Vec::<GeometryRenderInfo>::new();
        let (mut vertex_offset, mut index_offset) = (0u32, 0u32);

        resources
            .iter()
            .filter(|(_, geom)| !geom.has_materials())
            .for_each(|(tag, geom)| {
                log::info!("{} -> {}", tag, geom.aabb.extents());

                let geometry_handle = NonPbrRenderableHandle(geometry.len() as u32);

                geometry.push(GeometryRenderInfo {
                    vertex_offset,
                    index_offset,
                    index_count: geom.index_count(),
                    pbr_data_offset: 0,
                    pbr_data_range: 0,
                    nodes: geom.nodes().to_vec(),
                    aabb: geom.aabb,
                });

                vertex_offset += geom.vertex_count();
                index_offset += geom.index_count();

                handles.insert((*tag).clone(), geometry_handle);
            });

        let _vertex_bytes = vertex_offset as DeviceSize * size_of::<GeometryVertex>() as DeviceSize;
        let vertex_data: SmallVec<[&[GeometryVertex]; 8]> = resources
            .iter()
            .filter(|(_, geom)| !geom.has_materials())
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
        let indices_data: SmallVec<[&[u32]; 8]> = resources
            .iter()
            .filter(|(_, geom)| !geom.has_materials())
            .map(|(_, geom)| geom.indices())
            .collect();

        let index_buffer = UniqueBuffer::gpu_only_buffer(
            renderer,
            BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            MemoryPropertyFlags::DEVICE_LOCAL,
            &indices_data,
            None,
        )?;

        let geometries = geometry
            .into_iter()
            .map(|geom| NonPbrRenderable { geometry: geom })
            .collect::<Vec<_>>();

        Some(NonPbrResources {
            vertex_buffer,
            index_buffer,
            handles,
            geometries,
            pipeline_single: Self::create_non_pbr_rendering_pipeline(renderer, app_config)?,
            pipeline_instanced: Self::create_non_pbr_instanced_rendering_pipeline(
                renderer, app_config,
            )?,
        })
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

        imported_geometries
            .iter()
            .filter(|(_, geom)| geom.has_materials())
            .for_each(|(tag, geom)| {
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
                let pbr_data_offset = (size_of::<PbrMaterial>() as DeviceSize
                    * (pbr_data.len() as DeviceSize)) as u32;

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
            non_pbr: Self::create_non_pbr_resources(renderer, app_config, &imported_geometries)?,
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

    fn create_non_pbr_rendering_pipeline(
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
                        &app_config.engine.shader_path("nonpbr.frag.spv"),
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
                        &[*DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(ShaderStageFlags::FRAGMENT)],
                    )
                    .build(renderer.graphics_device())?,
                renderer.renderpass(),
                0,
            )
    }

    fn create_non_pbr_instanced_rendering_pipeline(
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
                        &app_config.engine.shader_path("nonpbr.frag.spv"),
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
                        &[
                            *DescriptorSetLayoutBinding::builder()
                                .binding(0)
                                .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::VERTEX),
                            *DescriptorSetLayoutBinding::builder()
                                .binding(1)
                                .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                                .descriptor_count(1)
                                .stage_flags(ShaderStageFlags::VERTEX),
                        ],
                    )
                    //
                    //set 1
                    .set(
                        1,
                        &[*DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(ShaderStageFlags::FRAGMENT)],
                    )
                    .build(renderer.graphics_device())?,
                renderer.renderpass(),
                0,
            )
    }
}

#[derive(Copy, Clone)]
pub struct MeshNode {
    pub parent: Option<u32>,
    pub name: SubmeshId,
    pub transform: nalgebra_glm::Mat4,
    pub aabb: AABB3,
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub index_count: u32,
}

impl MeshNode {
    pub fn new(value: &GeometryNode) -> Self {
        Self {
            parent: value.parent,
            name: (&value.name).into(),
            transform: value.transform,
            aabb: value.aabb,
            vertex_offset: value.vertex_offset,
            index_offset: value.index_offset,
            index_count: value.index_count,
        }
    }
}

#[derive(Clone, Default)]
struct MeshRenderInfo {
    name: MeshId,
    offset_vtx: u32,
    offset_idx: u32,
    vertices: u32,
    indices: u32,
    materials: u32,
    nodes: Vec<MeshNode>,
    bounds: AABB3,
    default_material: String,
}

#[derive(Copy, Clone, num_derive::FromPrimitive, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum BindlessResourceKind {
    UniformBufferGlobalTansform,
    StorageBufferInstanceData,
    StorageBufferMaterialDef,
    SamplerBaseColormap,
    SamplerMetallicRoughnessColormap,
    SamplerNormalMap,
    UniformBufferGlobalLight,
    SamplerEnvMapIrradiance,
    SamplerEnvMapPrefiltered,
    SamplerEnvMapBRDFLut,
    SamplerMiscTextures,
    SamplerMiscArrayTextures,
}

impl BindlessResourceKind {
    fn to_descriptor_type(&self) -> ash::vk::DescriptorType {
        match *self {
            BindlessResourceKind::UniformBufferGlobalTansform | BindlessResourceKind::UniformBufferGlobalLight => {
                DescriptorType::UNIFORM_BUFFER_DYNAMIC
            }
            BindlessResourceKind::StorageBufferInstanceData => DescriptorType::STORAGE_BUFFER_DYNAMIC,
            BindlessResourceKind::StorageBufferMaterialDef => DescriptorType::STORAGE_BUFFER,
            BindlessResourceKind::SamplerBaseColormap
            | BindlessResourceKind::SamplerMetallicRoughnessColormap
            | BindlessResourceKind::SamplerNormalMap
            | BindlessResourceKind::SamplerEnvMapBRDFLut
            | BindlessResourceKind::SamplerEnvMapPrefiltered
            | BindlessResourceKind::SamplerEnvMapIrradiance
            | BindlessResourceKind::SamplerMiscTextures
            | BindlessResourceKind::SamplerMiscArrayTextures => DescriptorType::COMBINED_IMAGE_SAMPLER,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct BindlessResourceHandle {
    rhandle: u64,
}

impl BindlessResourceHandle {
    pub fn create(id: BindlessResourceKind, tbl_idx: u32) -> Self {
        assert!(tbl_idx < 0xFF);
        Self {
            rhandle: ((id as u32) << 8) as u64 | (tbl_idx & 0xFF) as u64,
        }
    }

    pub fn descriptor_type(&self) -> BindlessResourceKind {
        use num::FromPrimitive;
        BindlessResourceKind::from_u8(((self.rhandle & 0xFF00) >> 8) as u8).unwrap()
    }

    pub fn handle(&self) -> u32 {
        (self.rhandle & 0xFF) as u32
    }
}

pub struct ResourceSystem {
    g_vertex_buffer: UniqueBuffer,
    g_index_buffer: UniqueBuffer,
    g_material_collection_buffer: UniqueBuffer,
    g_transforms_buffer: Cpu2GpuBuffer<GlobalTransforms>,
    g_lighting_buffer: Cpu2GpuBuffer<GlobalLightingData>,
    g_instances_buffer: Cpu2GpuBuffer<InstanceRenderInfo>,
    descriptor_pool: ash::vk::DescriptorPool,
    descriptor_sets: Vec<DescriptorSet>,
    meshes: HashMap<MeshId, MeshRenderInfo>,
    pipeline: UniqueGraphicsPipeline,
    materials: HashMap<String, u32>,
    samplers: SamplerResourceTable,
    resource_table: HashMap<BindlessResourceKind, Vec<UniqueImageWithView>>,
}

impl ResourceSystem {
    pub fn get_pbr_pipeline(&self) -> &UniqueGraphicsPipeline {
        &self.pipeline
    }

    fn default_sampler() -> SamplerCreateInfo {
        *SamplerCreateInfo::builder()
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
    }

    pub fn add_texture(
        &mut self,
        texture: UniqueImageWithView,
        id: BindlessResourceKind,
        sampler: Option<ash::vk::Sampler>,
        renderer: &VulkanRenderer,
    ) -> BindlessResourceHandle {
        let img_view = texture.image_view();

        if let Some(tbl) = self.resource_table.get_mut(&id) {
            tbl.push(texture);
        } else {
            self.resource_table.insert(id, vec![texture]);
        };

        let sampler =
            sampler.unwrap_or_else(|| self.get_sampler(&Self::default_sampler(), renderer));

        self.resource_table
            .get(&id)
            .map(|e| {
                let res_handle = BindlessResourceHandle::create(id, (e.len() - 1) as u32);

                unsafe {
                    renderer.graphics_device().update_descriptor_sets(
                        &[*WriteDescriptorSet::builder()
                            .descriptor_type(id.to_descriptor_type())
                            .dst_array_element(res_handle.handle())
                            .dst_binding(0)
                            .dst_set(self.descriptor_sets[id as usize])
                            .image_info(&[*DescriptorImageInfo::builder()
                                .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(img_view)
                                .sampler(sampler)])],
                        &[],
                    );
                }

                res_handle
            })
            .unwrap()
    }

    pub fn get_sampler(
        &mut self,
        ci: &SamplerCreateInfo,
        renderer: &VulkanRenderer,
    ) -> ash::vk::Sampler {
        self.samplers.get_sampler(ci, renderer)
    }

    pub fn get_material_id(&self, mtl: &str) -> u32 {
        *self
            .materials
            .get(mtl)
            .expect(&format!("material {mtl} not found"))
    }

    pub fn get_mesh_material(&self, mesh: MeshId) -> u32 {
        let mesh = self.get_mesh_info(mesh);
        self.get_material_id(&mesh.default_material)
    }

    pub fn get_mesh_info(&self, mesh: MeshId) -> &MeshRenderInfo {
        self.meshes
            .get(&mesh)
            .expect(&format!("Mesh {mesh} not found"))
    }

    pub fn get_submesh_info(&self, mesh_id: MeshId, submesh: SubmeshId) -> &MeshNode {
        let mesh = self.get_mesh_info(mesh_id);
        mesh.nodes
            .iter()
            .find(|node| node.name == submesh)
            .expect(&format!("Node {submesh} of mesh {mesh_id} not found"))
    }

    pub fn create(renderer: &VulkanRenderer, app_config: &AppConfig) -> Option<Self> {
        let s = Instant::now();

        let mut sampler_table = SamplerResourceTable::new();

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

        #[derive(Copy, Clone, Default)]
        struct Offsets {
            vertex: u32,
            index: u32,
            material: u32,
        }

        let mut materials = HashMap::<String, u32>::new();
        materials.insert("default_mtl".to_string(), 0);

        let mut meshes = HashMap::<MeshId, MeshRenderInfo>::new();

        let _offsets = imported_geometries.iter().fold(
            Offsets::default(),
            |off, (imp_geom_name, imp_geom_data)| {
                let m = MeshRenderInfo {
                    name: imp_geom_name.into(),
                    offset_vtx: off.vertex,
                    offset_idx: off.index,
                    vertices: imp_geom_data.vertex_count(),
                    indices: imp_geom_data.index_count(),
                    materials: imp_geom_data.pbr_materials().len() as u32,
                    nodes: imp_geom_data
                        .nodes()
                        .iter()
                        .map(|gnode| MeshNode::new(gnode))
                        .collect::<Vec<MeshNode>>(),
                    bounds: imp_geom_data.aabb,
                    default_material: "default_mtl".to_string(),
                };

                meshes.insert(imp_geom_name.into(), m);

                Offsets {
                    vertex: off.vertex + imp_geom_data.vertex_count(),
                    index: off.index + imp_geom_data.index_count(),
                    material: 0,
                }
            },
        );

        let vertex_data = imported_geometries
            .iter()
            .map(|(_, g)| g.vertices())
            .collect::<SmallVec<[&[GeometryVertex]; 8]>>();

        let g_vertex_buffer = UniqueBuffer::gpu_only_buffer(
            renderer,
            BufferUsageFlags::VERTEX_BUFFER,
            MemoryPropertyFlags::DEVICE_LOCAL,
            &vertex_data,
            None,
        )?;

        let index_data = imported_geometries
            .iter()
            .map(|(_, g)| g.indices())
            .collect::<SmallVec<[&[u32]; 8]>>();

        let g_index_buffer = UniqueBuffer::gpu_only_buffer(
            renderer,
            BufferUsageFlags::INDEX_BUFFER,
            MemoryPropertyFlags::DEVICE_LOCAL,
            &index_data,
            None,
        )?;

        //
        // load default materials
        let default_work_pkg = renderer.create_work_package()?;

        let mut base_color_textures = vec![UniqueImageWithView::from_ktx(
            renderer,
            &default_work_pkg,
            app_config.engine.texture_path("uv_grids/ash_uvgrid01.ktx2"),
        )
        .expect("Failed to load default grid texture")];

        let mut metallic_roughness_textures = vec![UniqueImageWithView::from_ktx(
            renderer,
            &default_work_pkg,
            app_config.engine.texture_path("uv_grids/ash_uvgrid01.ktx2"),
        )
        .expect("Failed to load default grid texture")];

        let mut normals_textures = vec![UniqueImageWithView::from_ktx(
            renderer,
            &default_work_pkg,
            app_config.engine.texture_path("uv_grids/ash_uvgrid01.ktx2"),
        )
        .expect("Failed to load default grid texture")];

        renderer.push_work_package(default_work_pkg);

        let mut material_collection = Vec::<PbrMaterial>::new();
        material_collection.push(PbrMaterial {
            base_color_factor: nalgebra_glm::vec4(1f32, 1f32, 1f32, 1f32),
            metallic_factor: 0f32,
            roughness_factor: 0f32,
            base_color_texarray_id: 0,
            metallic_rough_texarray_id: 0,
            normal_texarray_id: 0,
        });

        imported_geometries.iter().for_each(|(name, g)| {
            let mtl_coll_offset = material_collection.len() as u32;

            //
            // no materials, apply default
            if !g.has_materials() {
                return;
            }

            let (base_color_width, base_color_height, img_src) = g.pbr_base_color_images();

            use ash::vk::ImageCreateInfo;

            assert!(!g.pbr_materials().is_empty());

            let base_color_texarray_offset = base_color_textures.len() as u32;
            img_src.iter().for_each(|&bc_img| {
                let work_pkg = renderer
                    .create_work_package()
                    .expect("Failed to create work package");

                let img = UniqueImage::with_data(
                    renderer,
                    &ImageCreateInfo::builder()
                        .array_layers(1)
                        .format(Format::R8G8B8A8_SRGB)
                        .image_type(ImageType::TYPE_2D)
                        .initial_layout(ImageLayout::UNDEFINED)
                        .mip_levels(1)
                        .samples(SampleCountFlags::TYPE_1)
                        .sharing_mode(SharingMode::EXCLUSIVE)
                        .tiling(ImageTiling::OPTIMAL)
                        .extent(Extent3D {
                            width: base_color_width,
                            height: base_color_height,
                            depth: 1,
                        })
                        .usage(ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED),
                    &[bc_img],
                    &work_pkg,
                )
                .expect("Failed to create image");

                let img_view = UniqueImageView::from_image(renderer, &img)
                    .expect("Failed to create image view");

                base_color_textures.push(UniqueImageWithView(img, img_view));
                renderer.push_work_package(work_pkg);
            });

            let (mr_width, mr_height, mr_img_src) = g.pbr_metallic_roughness_images();
            let mr_texarray_offset = metallic_roughness_textures.len() as u32;

            mr_img_src.iter().for_each(|&img_src| {
                let work_pkg = renderer
                    .create_work_package()
                    .expect("Failed to create work package");
                let img = UniqueImage::with_data(
                    renderer,
                    &ImageCreateInfo::builder()
                        .array_layers(1)
                        .format(Format::R8G8B8A8_UNORM)
                        .image_type(ImageType::TYPE_2D)
                        .initial_layout(ImageLayout::UNDEFINED)
                        .mip_levels(1)
                        .samples(SampleCountFlags::TYPE_1)
                        .sharing_mode(SharingMode::EXCLUSIVE)
                        .tiling(ImageTiling::OPTIMAL)
                        .extent(Extent3D {
                            width: mr_width,
                            height: mr_height,
                            depth: 1,
                        })
                        .usage(ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED),
                    &[img_src],
                    &work_pkg,
                )
                .expect("Failed to create image");

                let img_view = UniqueImageView::from_image(renderer, &img)
                    .expect("Failed to create image view");

                metallic_roughness_textures.push(UniqueImageWithView(img, img_view));
                renderer.push_work_package(work_pkg);
            });

            let (nr_width, nr_height, nr_imgs) = g.pbr_normal_images();
            let nr_texarray_offset = normals_textures.len() as u32;

            normals_textures.extend(nr_imgs.iter().map(|&normal_img| {
                let work_pkg = renderer
                    .create_work_package()
                    .expect("Failed to create work package");
                let img = UniqueImage::with_data(
                    renderer,
                    &ImageCreateInfo::builder()
                        .array_layers(1)
                        .format(Format::R8G8B8A8_UNORM)
                        .image_type(ImageType::TYPE_2D)
                        .initial_layout(ImageLayout::UNDEFINED)
                        .mip_levels(1)
                        .samples(SampleCountFlags::TYPE_1)
                        .sharing_mode(SharingMode::EXCLUSIVE)
                        .tiling(ImageTiling::OPTIMAL)
                        .extent(Extent3D {
                            width: nr_width,
                            height: nr_height,
                            depth: 1,
                        })
                        .usage(ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED),
                    &[normal_img],
                    &work_pkg,
                )
                .expect("Failed to create image");

                let img_view = UniqueImageView::from_image(renderer, &img)
                    .expect("Failed to create image view");

                renderer.push_work_package(work_pkg);

                UniqueImageWithView(img, img_view)
            }));

            material_collection.extend(g.pbr_materials().iter().map(|&mtl| PbrMaterial {
                base_color_texarray_id: mtl.base_color_texarray_id + base_color_texarray_offset,
                metallic_rough_texarray_id: mtl.metallic_rough_texarray_id + mr_texarray_offset,
                normal_texarray_id: mtl.normal_texarray_id + nr_texarray_offset,
                ..mtl
            }));

            let material_name = format!("{name}_default");
            materials.insert(material_name.clone(), mtl_coll_offset);

            meshes.entry(name.into()).and_modify(|mesh_render_info| {
                mesh_render_info.default_material = material_name;
            });
        });

        let g_material_collection_buffer = UniqueBuffer::gpu_only_buffer(
            renderer,
            BufferUsageFlags::STORAGE_BUFFER,
            MemoryPropertyFlags::DEVICE_LOCAL,
            &[&material_collection],
            Some(renderer.device_properties().limits.non_coherent_atom_size),
        )
        .expect("Failed to create global material definition buffer");

        let g_transforms_buffer = Cpu2GpuBuffer::<GlobalTransforms>::create(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
            1,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let g_lighting_buffer = Cpu2GpuBuffer::<GlobalLightingData>::create(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
            1,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let descriptor_pool = unsafe {
            renderer.graphics_device().create_descriptor_pool(
                &DescriptorPoolCreateInfo::builder()
                    .max_sets(64)
                    .pool_sizes(&[
                        *DescriptorPoolSize::builder()
                            .ty(DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1024),
                        *DescriptorPoolSize::builder()
                            .ty(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                            .descriptor_count(1024),
                        *DescriptorPoolSize::builder()
                            .ty(DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1024),
                        *DescriptorPoolSize::builder()
                            .ty(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .descriptor_count(1024),
                    ]),
                None,
            )
        }
        .expect("Failed to create bindless descriptor pool");

        let layout_sbd = unsafe {
            let mut flags = DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&[DescriptorBindingFlags::PARTIALLY_BOUND]);
            renderer.graphics_device().create_descriptor_set_layout(
                &DescriptorSetLayoutCreateInfo::builder()
                    .push_next(&mut flags)
                    .bindings(&[*DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .descriptor_count(16)
                        .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                        .stage_flags(ShaderStageFlags::ALL)]),
                None,
            )
        }
        .expect("Failed to create set layout storage buffer dynamic");

        let layout_sb = unsafe {
            let mut flags = DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&[DescriptorBindingFlags::PARTIALLY_BOUND]);
            renderer.graphics_device().create_descriptor_set_layout(
                &DescriptorSetLayoutCreateInfo::builder()
                    .push_next(&mut flags)
                    .bindings(&[*DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .descriptor_count(64)
                        .descriptor_type(DescriptorType::STORAGE_BUFFER)
                        .stage_flags(ShaderStageFlags::ALL)]),
                None,
            )
        }
        .expect("Failed to create set layout storage buffer");

        let layout_cs = unsafe {
            let mut flags = DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&[DescriptorBindingFlags::PARTIALLY_BOUND]);
            renderer.graphics_device().create_descriptor_set_layout(
                &DescriptorSetLayoutCreateInfo::builder()
                    .push_next(&mut flags)
                    .bindings(&[*DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .descriptor_count(64)
                        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .stage_flags(ShaderStageFlags::ALL)]),
                None,
            )
        }
        .expect("Failed to create set layout combined image sampler");

        let layout_ubd = unsafe {
            let mut flags = DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&[DescriptorBindingFlags::PARTIALLY_BOUND]);
            renderer.graphics_device().create_descriptor_set_layout(
                &DescriptorSetLayoutCreateInfo::builder()
                    .push_next(&mut flags)
                    .bindings(&[*DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .descriptor_count(4)
                        .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .stage_flags(ShaderStageFlags::ALL)]),
                None,
            )
        }
        .expect("Failed to create set layout uniform buffer dynamic");

        let descriptor_set_layouts = vec![
            //
            // uniform buffer
            layout_ubd, //
            // dyn storage buffers VS
            layout_sbd, //
            // storage buffer FS
            layout_sb, layout_cs, layout_cs, layout_cs, layout_ubd, layout_cs, layout_cs,
            layout_cs, //
            // misc textures
            layout_cs, //
            // misc textures array
            layout_cs,
        ];

        let pipeline_layout = unsafe {
            renderer.graphics_device().create_pipeline_layout(
                &PipelineLayoutCreateInfo::builder()
                    .set_layouts(&descriptor_set_layouts)
                    .push_constant_ranges(&[*PushConstantRange::builder()
                        .stage_flags(ShaderStageFlags::ALL)
                        .offset(0)
                        .size(PushConstVertex::SIZE)]),
                None,
            )
        }
        .expect("Failed to create bindless pipeline layout");

        let pipeline = GraphicsPipelineBuilder::new()
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
                        &app_config.engine.shader_path("pbr.bindless.vert.spv"),
                    ),
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(
                        &app_config.engine.shader_path("pbr.bindless.frag.spv"),
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
                (pipeline_layout, descriptor_set_layouts),
                renderer.renderpass(),
                0,
            )?;

        let descriptor_sets = unsafe {
            renderer.graphics_device().allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(pipeline.descriptor_layouts()),
            )
        }
        .expect("Failed to allocate descriptor sets");

        log::info!("Allocated {} descriptor sets", descriptor_sets.len());

        let g_instances_buffer = Cpu2GpuBuffer::<InstanceRenderInfo>::create(
            renderer,
            BufferUsageFlags::STORAGE_BUFFER,
            renderer.device_properties().limits.non_coherent_atom_size,
            1024,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let default_sampler = sampler_table.get_sampler(
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
                .max_anisotropy(1f32),
            renderer,
        );

        log::info!(
            "Base color maps: {}, Metallic maps {}, Normal maps {}",
            base_color_textures.len(),
            metallic_roughness_textures.len(),
            normals_textures.len()
        );

        unsafe {
            renderer.graphics_device().update_descriptor_sets(
                &[
                    *WriteDescriptorSet::builder()
                        .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .dst_array_element(0)
                        .dst_binding(0)
                        .dst_set(
                            descriptor_sets[BindlessResourceKind::UniformBufferGlobalTansform as usize],
                        )
                        .buffer_info(&[*DescriptorBufferInfo::builder()
                            .buffer(g_transforms_buffer.buffer.buffer)
                            .offset(0)
                            .range(g_transforms_buffer.bytes_one_frame)]),
                    *WriteDescriptorSet::builder()
                        .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                        .dst_array_element(0)
                        .dst_binding(0)
                        .dst_set(descriptor_sets[BindlessResourceKind::StorageBufferInstanceData as usize])
                        .buffer_info(&[*DescriptorBufferInfo::builder()
                            .buffer(g_instances_buffer.buffer.buffer)
                            .offset(0)
                            .range(g_instances_buffer.bytes_one_frame)]),
                    *WriteDescriptorSet::builder()
                        .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .dst_array_element(0)
                        .dst_binding(0)
                        .dst_set(descriptor_sets[BindlessResourceKind::UniformBufferGlobalLight as usize])
                        .buffer_info(&[*DescriptorBufferInfo::builder()
                            .buffer(g_lighting_buffer.buffer.buffer)
                            .offset(0)
                            .range(g_lighting_buffer.bytes_one_frame)]),
                    *WriteDescriptorSet::builder()
                        .descriptor_type(DescriptorType::STORAGE_BUFFER)
                        .dst_array_element(0)
                        .dst_binding(0)
                        .dst_set(descriptor_sets[BindlessResourceKind::StorageBufferMaterialDef as usize])
                        .buffer_info(&[*DescriptorBufferInfo::builder()
                            .buffer(g_material_collection_buffer.buffer)
                            .offset(0)
                            .range(ash::vk::WHOLE_SIZE)]),
                    *WriteDescriptorSet::builder()
                        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .dst_set(descriptor_sets[BindlessResourceKind::SamplerBaseColormap as usize])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .image_info(
                            &base_color_textures
                                .iter()
                                .map(|base_color_tex| {
                                    DescriptorImageInfo::builder()
                                        .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                        .image_view(base_color_tex.image_view())
                                        .sampler(default_sampler)
                                        .build()
                                })
                                .collect::<Vec<_>>(),
                        ),
                    *WriteDescriptorSet::builder()
                        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .dst_set(
                            descriptor_sets
                                [BindlessResourceKind::SamplerMetallicRoughnessColormap as usize],
                        )
                        .dst_binding(0)
                        .dst_array_element(0)
                        .image_info(
                            &metallic_roughness_textures
                                .iter()
                                .map(|metallic_roughness_tex| {
                                    *DescriptorImageInfo::builder()
                                        .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                        .image_view(metallic_roughness_tex.image_view())
                                        .sampler(default_sampler)
                                })
                                .collect::<Vec<_>>(),
                        ),
                    *WriteDescriptorSet::builder()
                        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .dst_set(descriptor_sets[BindlessResourceKind::SamplerNormalMap as usize])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .image_info(
                            &normals_textures
                                .iter()
                                .map(|normal_tex| {
                                    *DescriptorImageInfo::builder()
                                        .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                        .image_view(normal_tex.image_view())
                                        .sampler(default_sampler)
                                })
                                .collect::<Vec<_>>(),
                        ),
                ],
                &[],
            );
        }

        Some(ResourceSystem {
            g_vertex_buffer,
            g_index_buffer,
            g_material_collection_buffer,
            g_transforms_buffer,
            g_lighting_buffer,
            g_instances_buffer,
            descriptor_pool,
            descriptor_sets,
            meshes,
            pipeline,
            materials,
            samplers: sampler_table,
            resource_table: [
                (BindlessResourceKind::SamplerBaseColormap, base_color_textures),
                (
                    BindlessResourceKind::SamplerMetallicRoughnessColormap,
                    metallic_roughness_textures,
                ),
                (BindlessResourceKind::SamplerNormalMap, normals_textures),
            ]
            .into(),
        })
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
struct GlobalTransforms {
    projection_view: nalgebra_glm::Mat4,
    view: nalgebra_glm::Mat4,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct GlobalLightingData {
    eye_pos: nalgebra_glm::Vec3,
    skybox: u32,
}

#[derive(Copy, Clone)]
#[repr(C, align(16))]
struct InstanceRenderInfo {
    model: nalgebra_glm::Mat4,
    mtl_coll_offset: u32,
}

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct PushConstVertex {
    pub model: nalgebra_glm::Mat4,
    pub atlas_id: u32,
}

impl PushConstVertex {
    pub const SIZE: u32 = std::mem::size_of::<Self>() as u32;
}

impl std::convert::AsRef<[u8]> for PushConstVertex {
    fn as_ref(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self as *const _ as *const u8, std::mem::size_of::<Self>())
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct MeshId(u64);

impl std::default::Default for MeshId {
    fn default() -> Self {
        Self::from(&"")
    }
}

impl<T: AsRef<str>> std::convert::From<T> for MeshId {
    fn from(value: T) -> Self {
        let mut hasher = fnv::FnvHasher::default();
        use std::hash::Hasher;
        hasher.write(value.as_ref().as_bytes());

        MeshId(hasher.finish())
    }
}

impl std::fmt::Display for MeshId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MeshId ({})", self.0)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct SubmeshId(u64);

impl<T: AsRef<str>> std::convert::From<T> for SubmeshId {
    fn from(value: T) -> Self {
        let mut hasher = fnv::FnvHasher::default();
        use std::hash::Hasher;
        hasher.write(value.as_ref().as_bytes());

        SubmeshId(hasher.finish())
    }
}

impl std::fmt::Display for SubmeshId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SubmeshId ({})", self.0)
    }
}

struct DrawRequest {
    mesh: MeshId,
    submesh: Option<SubmeshId>,
    material: Option<String>,
    obj2world: nalgebra_glm::Mat4,
}

pub struct DrawingSys {
    drawcalls_buffer: Cpu2GpuBuffer<ash::vk::DrawIndexedIndirectCommand>,
    requests: Vec<DrawRequest>,
}

impl DrawingSys {
    pub fn create(renderer: &VulkanRenderer) -> Option<Self> {
        let drawcalls_buffer = Cpu2GpuBuffer::<ash::vk::DrawIndexedIndirectCommand>::create(
            renderer,
            BufferUsageFlags::INDIRECT_BUFFER,
            renderer.device_properties().limits.non_coherent_atom_size,
            128,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        Some(Self {
            drawcalls_buffer,
            requests: Vec::with_capacity(1024),
        })
    }

    pub fn add_mesh(
        &mut self,
        mesh: MeshId,
        submesh: Option<SubmeshId>,
        mat_id: Option<String>,
        transform: &nalgebra_glm::Mat4,
    ) {
        self.requests.push(DrawRequest {
            mesh,
            submesh,
            material: mat_id,
            obj2world: *transform,
        });
    }

    pub fn setup_bindless(&self, skybox_id: u32, draw_ctx: &DrawContext) {
        draw_ctx
            .rsys
            .g_transforms_buffer
            .map_for_frame(draw_ctx.renderer, draw_ctx.frame_id as DeviceSize)
            .map(|mut transforms_buffer| {
                let transforms = GlobalTransforms {
                    projection_view: draw_ctx.projection_view,
                    view: draw_ctx.view_matrix,
                };

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &transforms as *const _,
                        transforms_buffer.as_mut_ptr() as *mut GlobalTransforms,
                        1,
                    );
                }
            });

        draw_ctx
            .rsys
            .g_lighting_buffer
            .map_for_frame(draw_ctx.renderer, draw_ctx.frame_id as DeviceSize)
            .map(|mut light_buffer| {
                let light_data = GlobalLightingData {
                    eye_pos: draw_ctx.cam_position,
                    skybox: skybox_id,
                };

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &light_data as *const _,
                        light_buffer.as_mut_ptr() as *mut GlobalLightingData,
                        1,
                    );
                }
            });

        // 4 -> ubo tf globals, 0 0 0
        // 64
        // 4 -> ubo light , 0, 0, 0
        use std::iter::{once, repeat};
        let bindig_offsets = once(
            draw_ctx
                .rsys
                .g_transforms_buffer
                .offset_for_frame(draw_ctx.frame_id as DeviceSize) as u32,
        )
        .chain(repeat(0u32).take(3))
        .chain(once(
            draw_ctx
                .rsys
                .g_instances_buffer
                .offset_for_frame(draw_ctx.frame_id as DeviceSize) as u32,
        ))
        .chain(repeat(0u32).take(15))
        .chain(once(
            draw_ctx
                .rsys
                .g_lighting_buffer
                .offset_for_frame(draw_ctx.frame_id as DeviceSize) as u32,
        ))
        .chain(repeat(0u32).take(3))
        .collect::<Vec<_>>();

        unsafe {
            draw_ctx
                .renderer
                .graphics_device()
                .cmd_bind_descriptor_sets(
                    draw_ctx.cmd_buff,
                    PipelineBindPoint::GRAPHICS,
                    draw_ctx.rsys.pipeline.layout,
                    0,
                    &draw_ctx.rsys.descriptor_sets,
                    &bindig_offsets,
                );
        }
    }

    pub fn draw(&mut self, draw_ctx: &DrawContext) {
        type MeshWithSubmeshPair = (MeshId, Option<SubmeshId>);

        let mut mesh_groups = HashMap::<MeshWithSubmeshPair, Vec<InstanceRenderInfo>>::new();

        self.requests.iter().for_each(|rq| {
            let mtl_offset = rq
                .material
                .as_ref()
                .map(|rq_mtl| draw_ctx.rsys.get_material_id(&rq_mtl))
                .unwrap_or_else(|| draw_ctx.rsys.get_mesh_material(rq.mesh));

            let inst_render_info = InstanceRenderInfo {
                model: rq.obj2world,
                mtl_coll_offset: mtl_offset,
            };

            mesh_groups
                .entry((rq.mesh, rq.submesh))
                .and_modify(|e| {
                    e.push(inst_render_info);
                })
                .or_insert(vec![inst_render_info]);
        });

        let mut instance_offset = 0u32;
        let mut instance_data = Vec::<InstanceRenderInfo>::new();
        let mut drawcalls = Vec::<DrawIndexedIndirectCommand>::new();

        mesh_groups
            .into_iter()
            .for_each(|((mesh_id, maybe_submesh_id), instances)| {
                let mesh = draw_ctx.rsys.get_mesh_info(mesh_id);

                let (index_count, mesh_index_offset) = maybe_submesh_id
                    .map(|submesh_id| {
                        let si = draw_ctx.rsys.get_submesh_info(mesh_id, submesh_id);
                        (si.index_count, mesh.offset_idx + si.index_offset)
                    })
                    .unwrap_or_else(|| (mesh.indices, mesh.offset_idx));

                let drawcall = DrawIndexedIndirectCommand {
                    vertex_offset: 0,
                    index_count,
                    instance_count: instances.len() as u32,
                    first_index: mesh_index_offset,
                    first_instance: instance_offset,
                };

                instance_offset += instances.len() as u32;
                instance_data.extend(instances);
                drawcalls.push(drawcall);
            });

        draw_ctx
            .rsys
            .g_instances_buffer
            .map_for_frame(draw_ctx.renderer, draw_ctx.frame_id as DeviceSize)
            .map(|mut instance_buffer| unsafe {
                std::ptr::copy_nonoverlapping(
                    instance_data.as_ptr(),
                    instance_buffer.as_mut_ptr() as *mut InstanceRenderInfo,
                    instance_data.len(),
                );
            });

        self.drawcalls_buffer
            .map_for_frame(draw_ctx.renderer, draw_ctx.frame_id as DeviceSize)
            .map(|mut drawcalls_buffer| unsafe {
                std::ptr::copy_nonoverlapping(
                    drawcalls.as_ptr(),
                    drawcalls_buffer.as_mut_ptr() as *mut DrawIndexedIndirectCommand,
                    drawcalls.len(),
                );
            });

        let dc = draw_ctx.renderer.graphics_device();

        unsafe {
            dc.cmd_bind_pipeline(
                draw_ctx.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                draw_ctx.rsys.pipeline.pipeline,
            );

            dc.cmd_bind_vertex_buffers(
                draw_ctx.cmd_buff,
                0,
                &[draw_ctx.rsys.g_vertex_buffer.buffer],
                &[0],
            );

            dc.cmd_bind_index_buffer(
                draw_ctx.cmd_buff,
                draw_ctx.rsys.g_index_buffer.buffer,
                0,
                IndexType::UINT32,
            );
        }

        //
        // draw skybox first

        unsafe {
            dc.cmd_bind_pipeline(
                draw_ctx.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                draw_ctx.rsys.pipeline.pipeline,
            );
            dc.cmd_set_viewport(draw_ctx.cmd_buff, 0, &[draw_ctx.viewport]);
            dc.cmd_set_scissor(draw_ctx.cmd_buff, 0, &[draw_ctx.scissor]);

            dc.cmd_draw_indexed_indirect(
                draw_ctx.cmd_buff,
                self.drawcalls_buffer.buffer.buffer,
                self.drawcalls_buffer
                    .offset_for_frame(draw_ctx.frame_id as DeviceSize),
                drawcalls.len() as u32,
                std::mem::size_of::<DrawIndexedIndirectCommand>() as u32,
            );
        }

        //
        // draw ui

        self.requests.clear();
    }
}

#[derive(Copy, Clone)]
pub struct SamplerDescription(ash::vk::SamplerCreateInfo);

impl std::ops::Deref for SamplerDescription {
    type Target = ash::vk::SamplerCreateInfo;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SamplerDescription {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl std::convert::AsRef<ash::vk::SamplerCreateInfo> for SamplerDescription {
    fn as_ref(&self) -> &ash::vk::SamplerCreateInfo {
        &self.0
    }
}

impl std::convert::AsMut<ash::vk::SamplerCreateInfo> for SamplerDescription {
    fn as_mut(&mut self) -> &mut ash::vk::SamplerCreateInfo {
        &mut self.0
    }
}

impl std::convert::From<ash::vk::SamplerCreateInfo> for SamplerDescription {
    fn from(value: ash::vk::SamplerCreateInfo) -> Self {
        Self(value)
    }
}

impl std::hash::Hash for SamplerDescription {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.address_mode_u.hash(state);
        self.address_mode_v.hash(state);
        self.address_mode_w.hash(state);
        self.anisotropy_enable.hash(state);
        self.border_color.hash(state);
        self.compare_enable.hash(state);
        self.compare_op.hash(state);
        self.flags.hash(state);
        self.mag_filter.hash(state);
        self.unnormalized_coordinates.hash(state);

        use crate::math::integer_decode;
        integer_decode(self.max_anisotropy as f64).hash(state);
        integer_decode(self.max_lod as f64).hash(state);
        integer_decode(self.min_lod as f64).hash(state);
        self.min_filter.hash(state);
        integer_decode(self.mip_lod_bias as f64).hash(state);
        self.mipmap_mode.hash(state);
    }
}

impl std::cmp::PartialEq for SamplerDescription {
    fn eq(&self, rhs: &Self) -> bool {
        self.address_mode_u == rhs.address_mode_u
            && self.address_mode_v == rhs.address_mode_v
            && self.address_mode_w == rhs.address_mode_w
            && self.anisotropy_enable == rhs.anisotropy_enable
            && self.border_color == rhs.border_color
            && self.compare_enable == rhs.compare_enable
            && self.compare_op == rhs.compare_op
            && self.mag_filter == rhs.mag_filter
            && self.max_anisotropy == rhs.max_anisotropy
            && self.max_lod == rhs.max_lod
            && self.min_lod == rhs.min_lod
            && self.min_filter == rhs.min_filter
            && self.mip_lod_bias == rhs.mip_lod_bias
            && self.mipmap_mode == rhs.mipmap_mode
            && self.unnormalized_coordinates == rhs.unnormalized_coordinates
    }
}

impl std::cmp::Eq for SamplerDescription {}

struct SamplerResourceTable {
    tbl: HashMap<SamplerDescription, UniqueSampler>,
}

impl SamplerResourceTable {
    pub fn new() -> Self {
        Self {
            tbl: HashMap::new(),
        }
    }

    pub fn get_sampler(
        &mut self,
        ci: &SamplerCreateInfo,
        renderer: &VulkanRenderer,
    ) -> ash::vk::Sampler {
        let sd: SamplerDescription = (*ci).into();

        if let Some(existing_sampler) = self.tbl.get(&sd) {
            existing_sampler.sampler
        } else {
            let us = UniqueSampler::new(renderer.graphics_device(), ci)
                .expect(&format!("Failed to create sampler: {:?}", ci));
            let res = us.sampler;
            self.tbl.insert(sd, us);
            res
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PipelineLayoutDescription<'a> {
    pub layout: PipelineLayout,
    pub descriptor_sets_layouts: &'a [DescriptorSetLayout],
}
