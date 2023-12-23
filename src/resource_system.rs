use ash::vk::{
    BufferUsageFlags, CullModeFlags, DescriptorSet, DynamicState, Extent3D, Filter, Format,
    FrontFace, Handle, ImageLayout, ImageTiling, ImageType, ImageUsageFlags, ObjectType,
    PipelineLayout, PipelineRasterizationStateCreateInfo, PolygonMode, SampleCountFlags,
    SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, ShaderStageFlags, SharingMode,
    VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
};
use chrono::Duration;
use memoffset::offset_of;

use nalgebra_glm as glm;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::{collections::HashMap, mem::size_of, path::Path, time::Instant};

use crate::{
    app_config::AppConfig,
    bindless::{BindlessResourceHandle, BindlessResourceSystem},
    imported_geometry::{GeometryNode, GeometryVertex, ImportedGeometry},
    math::AABB3,
    pbr::PbrMaterial,
    vk_renderer::{
        BindlessPipeline, GraphicsPipelineBuilder, ShaderModuleDescription, ShaderModuleSource,
        UniqueBuffer, UniqueImage, UniqueImageView, UniqueImageWithView, UniqueSampler,
        VulkanRenderer,
    },
    ProgramError,
};

#[derive(Copy, Clone)]
#[repr(C)]
pub struct InstanceRenderInfo {
    pub model: nalgebra_glm::Mat4,
    pub mtl_coll_offset: u32,
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
pub struct MeshRenderInfo {
    pub name: MeshId,
    pub offset_vtx: u32,
    pub offset_idx: u32,
    pub vertices: u32,
    pub indices: u32,
    pub materials: u32,
    pub nodes: Vec<MeshNode>,
    pub bounds: AABB3,
    pub default_material: String,
}

impl MeshRenderInfo {
    pub fn get_node(&self, name: SubmeshId) -> &MeshNode {
        self.nodes
            .iter()
            .find(|node| node.name == name)
            .expect(&format!("Mesh {}, node {name} not found", self.name))
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum EffectType {
    Pbr,
    BasicEmissive,
}

#[derive(Copy, Clone)]
pub struct MissileSmokePoint {
    pub pt: glm::Vec3,
}

pub struct CachedTexture {
    pub img: UniqueImageWithView,
    pub handle: BindlessResourceHandle,
}

pub struct ResourceSystem {
    pub g_vertex_buffer: UniqueBuffer,
    pub g_index_buffer: UniqueBuffer,
    g_material_collection_buffer: UniqueBuffer,
    pub material_buffer: BindlessResourceHandle,
    meshes: HashMap<MeshId, MeshRenderInfo>,
    materials: HashMap<String, u32>,
    samplers: SamplerResourceTable,
    effect_table: HashMap<EffectType, BindlessPipeline>,
    pub bindless: BindlessResourceSystem,
    textures: HashMap<String, CachedTexture>,
}

impl ResourceSystem {
    pub fn get_effect(&self, e: EffectType) -> &BindlessPipeline {
        self.effect_table
            .get(&e)
            .expect(&format!("Effect {:?} not found", e))
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

    pub fn add_texture_bindless(
        &mut self,
        id: &str,
        renderer: &VulkanRenderer,
        img: UniqueImageWithView,
        sampler_info: Option<SamplerDescription>,
    ) -> BindlessResourceHandle {
        let si = sampler_info
            .map(|s| *s)
            .unwrap_or_else(|| Self::default_sampler());

        let sampler = self.samplers.get_sampler(&si, renderer);
        let handle = self
            .bindless
            .register_image(renderer, img.image_view(), sampler);
        self.textures
            .insert(id.to_string(), CachedTexture { img, handle });
        handle
    }

    pub fn get_texture(&mut self, id: &str) -> BindlessResourceHandle {
        self.textures
            .get(id)
            .map(|cached_tex| cached_tex.handle)
            .expect(&format!("Texture {id} not found"))
    }

    pub fn bindless_setup(&self) -> BindlessSetup {
        BindlessSetup {
            pipeline_layout: self.bindless.bindless_pipeline_layout(),
            descriptor_set: self.bindless.descriptor_sets(),
        }
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

    pub fn create(renderer: &VulkanRenderer, app_config: &AppConfig) -> Result<Self, ProgramError> {
        let s = Instant::now();

        let mut bindless =
            BindlessResourceSystem::new(renderer).expect("Failed to create bindless system");
        let mut textures = HashMap::<String, CachedTexture>::new();

        let mut sampler_table = SamplerResourceTable::new();
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

        let g_vertex_buffer =
            UniqueBuffer::gpu_only_buffer(renderer, BufferUsageFlags::VERTEX_BUFFER, &vertex_data)?;

        renderer.debug_set_object_tag("Global mesh buffer", &g_vertex_buffer);

        let index_data = imported_geometries
            .iter()
            .map(|(_, g)| g.indices())
            .collect::<SmallVec<[&[u32]; 8]>>();

        let g_index_buffer =
            UniqueBuffer::gpu_only_buffer(renderer, BufferUsageFlags::INDEX_BUFFER, &index_data)?;

        renderer.debug_set_object_tag("Global index buffer", &g_index_buffer);

        //
        // load default/null texture
        let default_work_pkg = renderer
            .create_work_package()
            .expect("Failed to create work package");

        let null_texture = UniqueImageWithView::from_ktx(
            renderer,
            &default_work_pkg,
            app_config.engine.texture_path("uv_grids/ash_uvgrid01.ktx2"),
        )
        .expect("Oopsie");

        let null_texture_handle =
            bindless.register_image(renderer, null_texture.image_view(), default_sampler);

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

            img_src.iter().enumerate().for_each(|(i, &bc_img)| {
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

                let texture_and_view = UniqueImageWithView(img, img_view);
                let texture_handle = bindless.register_image(
                    renderer,
                    texture_and_view.image_view(),
                    default_sampler,
                );
                textures.insert(
                    format!("{name}/mtl_basecolor_{i}"),
                    CachedTexture {
                        img: texture_and_view,
                        handle: texture_handle,
                    },
                );
                renderer.push_work_package(work_pkg);
            });

            let (mr_width, mr_height, mr_img_src) = g.pbr_metallic_roughness_images();

            mr_img_src.iter().enumerate().for_each(|(i, &img_src)| {
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

                let texture_and_view = UniqueImageWithView(img, img_view);

                let texture_handle = bindless.register_image(
                    renderer,
                    texture_and_view.image_view(),
                    default_sampler,
                );
                textures.insert(
                    format!("{name}/mtl_metallic_roughness_{i}"),
                    CachedTexture {
                        img: texture_and_view,
                        handle: texture_handle,
                    },
                );

                renderer.push_work_package(work_pkg);
            });

            let (nr_width, nr_height, nr_imgs) = g.pbr_normal_images();

            nr_imgs.iter().enumerate().for_each(|(i, &normal_img)| {
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

                let texture_and_view = UniqueImageWithView(img, img_view);

                let texture_handle = bindless.register_image(
                    renderer,
                    texture_and_view.image_view(),
                    default_sampler,
                );

                textures.insert(
                    format!("{name}/mtl_normal_{i}"),
                    CachedTexture {
                        img: texture_and_view,
                        handle: texture_handle,
                    },
                );

                renderer.push_work_package(work_pkg);
            });

            material_collection.extend(g.pbr_materials().iter().map(|&mtl| {
                let base_color = textures
                    .get(&format!(
                        "{name}/mtl_basecolor_{}",
                        mtl.base_color_texarray_id
                    ))
                    .expect("oopsie");

                let metallic = textures
                    .get(&format!(
                        "{name}/mtl_metallic_roughness_{}",
                        mtl.metallic_rough_texarray_id
                    ))
                    .expect("Oopsie");

                let normal = textures
                    .get(&format!("{name}/mtl_normal_{}", mtl.normal_texarray_id))
                    .expect("Oppsie");

                PbrMaterial {
                    base_color_texarray_id: base_color.handle.handle(),
                    metallic_rough_texarray_id: metallic.handle.handle(),
                    normal_texarray_id: normal.handle.handle(),
                    ..mtl
                }
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
            &[&material_collection],
        )?;

        renderer.debug_set_object_tag(
            "PBR material definition SSBO",
            &g_material_collection_buffer,
        );

        let material_buffer = bindless.register_ssbo(renderer, &g_material_collection_buffer);

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
            .build_bindless(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                bindless.bindless_pipeline_layout(),
                renderer.renderpass(),
                0,
            )?;

        let emissive_effect = Self::create_emissive_effect(
            renderer,
            app_config,
            bindless.bindless_pipeline_layout(),
        )?;

        Ok(ResourceSystem {
            g_vertex_buffer,
            material_buffer,
            g_index_buffer,
            g_material_collection_buffer,
            meshes,
            materials,
            samplers: sampler_table,
            effect_table: [
                (EffectType::Pbr, pipeline),
                (EffectType::BasicEmissive, emissive_effect),
            ]
            .into(),
            bindless,
            textures,
        })
    }

    pub fn add_missile_some(&mut self, segments: &[MissileSmokePoint]) {}

    fn create_emissive_effect(
        renderer: &VulkanRenderer,
        app_config: &AppConfig,
        layout: PipelineLayout,
    ) -> Result<BindlessPipeline, ProgramError> {
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
                        &app_config.engine.shader_path("pbr.bindless.vert.spv"),
                    ),
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(
                        &app_config.engine.shader_path("emissive.frag.spv"),
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
            .build_bindless(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                layout,
                renderer.renderpass(),
                0,
            )
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
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

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
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

#[derive(Copy, Clone)]
pub struct SamplerDescription(pub ash::vk::SamplerCreateInfo);

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

pub struct BindlessSetup<'a> {
    pub pipeline_layout: ash::vk::PipelineLayout,
    pub descriptor_set: &'a [DescriptorSet],
}
