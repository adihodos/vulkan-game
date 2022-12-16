use ash::vk::{BufferUsageFlags, DeviceSize, MemoryPropertyFlags, WHOLE_SIZE};
use chrono::Duration;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::{
    collections::HashMap, mem::size_of, path::Path, ptr::copy_nonoverlapping, time::Instant,
};

use crate::{
    app_config::{self, AppConfig},
    imported_geometry::{GeometryVertex, ImportedGeometry},
    pbr::{PbrMaterial, PbrMaterialTextureCollection},
    vk_renderer::{
        ScopedBufferMapping, UniqueBuffer, UniqueGraphicsPipeline, UniqueImageWithView,
        VulkanRenderer,
    },
};

#[derive(Copy, Clone, Debug, Default)]
pub struct RenderableGeometry {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub pbr_data_offset: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
pub struct RenderableGeometryHandle(u32);

pub struct ResourceHolder {
    vertex_buffer: UniqueBuffer,
    index_buffer: UniqueBuffer,
    pbr_data_buffer: UniqueBuffer,
    // pbr_materials: Vec<PbrMaterial>,
    pbr_textures: Vec<PbrMaterialTextureCollection>,
    // pipeline: UniqueGraphicsPipeline,
    geometry: Vec<RenderableGeometry>,
    handles: HashMap<String, RenderableGeometryHandle>,
}

impl ResourceHolder {
    pub fn get_geometry_handle(&self, name: &str) -> RenderableGeometryHandle {
        *self.handles.get(name).unwrap()
    }

    pub fn get_renderable_geometry(&self, handle: RenderableGeometryHandle) -> &RenderableGeometry {
        &self.geometry[handle.0 as usize]
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
            "Loaded geometries in {}m {}s {}ms",
            e.num_minutes(),
            e.num_seconds(),
            e.num_milliseconds()
        );

        let mut handles = HashMap::<String, RenderableGeometryHandle>::new();
        let mut pbr_data = Vec::<PbrMaterial>::new();
        let mut pbr_textures = Vec::<PbrMaterialTextureCollection>::new();
        let mut geometry = Vec::<RenderableGeometry>::new();
        let (mut vertex_offset, mut index_offset) = (0u32, 0u32);
        let pbr_data_aligned_size = VulkanRenderer::aligned_size_of_type::<PbrMaterial>(
            renderer.device_properties().limits.non_coherent_atom_size,
        );

        imported_geometries.iter().for_each(|(tag, geom)| {
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

            let geometry_handle = RenderableGeometryHandle(geometry.len() as u32);
            let pbr_data_offset = (pbr_data_aligned_size * (pbr_data.len() as DeviceSize)) as u32;

            pbr_textures.push(pbr_mtl_tex);
            geometry.push(RenderableGeometry {
                vertex_offset,
                index_offset,
                index_count: geom.index_count(),
                pbr_data_offset,
            });
            pbr_data.extend(geom.pbr_materials().iter());

            vertex_offset += geom.vertex_count();
            index_offset += geom.index_count();

            handles.insert((*tag).clone(), geometry_handle);
        });

        let vertex_bytes = vertex_offset as DeviceSize * size_of::<GeometryVertex>() as DeviceSize;
        let vertex_data: SmallVec<[&[GeometryVertex]; 8]> = imported_geometries
            .iter()
            .map(|(_, geom)| geom.vertices())
            .collect();

        let vertex_buffer = UniqueBuffer::gpu_only_buffer(
            renderer,
            BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            MemoryPropertyFlags::DEVICE_LOCAL,
            &vertex_data,
        )?;

        let index_bytes = index_offset as DeviceSize * size_of::<u32>() as DeviceSize;
        let indices_data: SmallVec<[&[u32]; 8]> = imported_geometries
            .iter()
            .map(|(_, geom)| geom.indices())
            .collect();

        let index_buffer = UniqueBuffer::gpu_only_buffer(
            renderer,
            BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
            MemoryPropertyFlags::DEVICE_LOCAL,
            &indices_data,
        )?;

        let pbr_bytes = pbr_data_aligned_size * pbr_data.len() as DeviceSize;
        let pbr_data_buffer = UniqueBuffer::gpu_only_buffer(
            renderer,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            MemoryPropertyFlags::DEVICE_LOCAL,
            &[&pbr_data],
        )?;

        Some(ResourceHolder {
            vertex_buffer,
            index_buffer,
            pbr_data_buffer,
            pbr_textures,
            geometry,
            handles,
        })
    }
}
