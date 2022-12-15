use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    mem::size_of,
};

use crate::pbr::PbrMaterial;
use ash::vk::DeviceSize;
use gltf::{
    buffer::{self, Data},
    image::{self, Source},
    mesh::Mode,
    scene::Transform,
    Document, Node,
};
use log::{error, info};
use mmap_rs::MmapOptions;
use nalgebra_glm::{identity, quat, translate, Mat4, Qua, Vec2, Vec3, Vec4};
use rayon::prelude::*;
use slice_of_array::prelude::*;

use nalgebra_glm as glm;

use crate::vk_renderer::ImageCopySource;

pub struct GeometryNode {
    pub parent: Option<u32>,
    pub name: String,
    pub transform: Mat4,
}

impl std::default::Default for GeometryNode {
    fn default() -> Self {
        GeometryNode {
            parent: None,
            name: String::new(),
            transform: glm::identity::<f32, 4>(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GeometryVertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub color: Vec4,
    pub tangent: Vec4,
    pub pbr_buf_id: u32,
}

impl std::default::Default for GeometryVertex {
    fn default() -> Self {
        GeometryVertex {
            pos: Vec3::new(0f32, 0f32, 0f32),
            normal: Vec3::new(0f32, 0f32, 0f32),
            uv: Vec2::new(0f32, 0f32),
            color: Vec4::new(0f32, 0f32, 0f32, 1f32),
            tangent: Vec4::new(0f32, 0f32, 0f32, 0f32),
            pbr_buf_id: 0u32, // ..Default::default()
        }
    }
}

#[derive(Debug)]
struct MaterialDef {
    name: String,
    base_color_src: u32,
    metallic_src: u32,
    normal_src: u32,
    base_color_factor: Vec4,
    metallic_factor: f32,
    roughness_factor: f32,
}

pub struct ImportedGeometry {
    nodes: Vec<GeometryNode>,
    vertices: Vec<GeometryVertex>,
    indices: Vec<u32>,
    buffers: Vec<buffer::Data>,
    images: Vec<image::Data>,
    gltf_mat_2_pbr_mat_mapping: HashMap<String, u32>,
    pbr_materials: Vec<PbrMaterial>,
    pixels_base_color: Vec<(u32, u32)>,
    pixels_metallic_roughness: Vec<(u32, u32)>,
    pixels_normal: Vec<(u32, u32)>,
}

impl ImportedGeometry {
    pub fn pbr_materials(&self) -> &[PbrMaterial] {
        &self.pbr_materials
    }

    fn get_image_pixels(&self, pixels: &[(u32, u32)]) -> Vec<ImageCopySource> {
        pixels
            .iter()
            .map(|(_, img_idx)| {
                let img = &self.images[*img_idx as usize];
                assert!(img.format == gltf::image::Format::R8G8B8A8);
                ImageCopySource {
                    src: img.pixels.as_ptr(),
                    bytes: (img.width * img.height * 4) as DeviceSize,
                }
            })
            .collect()
    }

    pub fn pbr_base_color_images(&self) -> (u32, u32, Vec<ImageCopySource>) {
        let img = &self.images[self.pixels_base_color[0].1 as usize];

        (
            img.width,
            img.height,
            self.get_image_pixels(&self.pixels_base_color),
        )
    }

    pub fn pbr_metallic_roughness_images(&self) -> (u32, u32, Vec<ImageCopySource>) {
        let img = &self.images[self.pixels_metallic_roughness[0].1 as usize];

        (
            img.width,
            img.height,
            self.get_image_pixels(&self.pixels_metallic_roughness),
        )
    }

    pub fn pbr_normal_images(&self) -> (u32, u32, Vec<ImageCopySource>) {
        let img = &self.images[self.pixels_metallic_roughness[0].1 as usize];

        (
            img.width,
            img.height,
            self.get_image_pixels(&self.pixels_normal),
        )
    }

    pub fn nodes(&self) -> &[GeometryNode] {
        &self.nodes
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    pub fn vertex_bytes(&self) -> usize {
        self.vertex_count() as usize * size_of::<GeometryVertex>()
    }

    pub fn index_bytes(&self) -> usize {
        self.index_count() as usize * size_of::<u32>()
    }

    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn vertices(&self) -> &[GeometryVertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    fn process_materials(&mut self, gltf_doc: &gltf::Document) {
        let materials = gltf_doc
            .materials()
            .map(|mtl| {
                let name = mtl
                    .name()
                    .expect("Unnamed materials are not supported!")
                    .to_string();
                let base_color_src = mtl
                    .pbr_metallic_roughness()
                    .base_color_texture()
                    .expect("Base color texture missing")
                    .texture()
                    .source()
                    .index() as u32;

                let metalic_roughness_src = mtl
                    .pbr_metallic_roughness()
                    .metallic_roughness_texture()
                    .expect("Missing metallic roughness texture")
                    .texture()
                    .source()
                    .index() as u32;

                let normal_src = mtl
                    .normal_texture()
                    .expect("Normal texture missing")
                    .texture()
                    .source()
                    .index() as u32;

                MaterialDef {
                    name,
                    base_color_src,
                    metallic_src: metalic_roughness_src,
                    normal_src,
                    base_color_factor: Vec4::from_row_slice(
                        &mtl.pbr_metallic_roughness().base_color_factor(),
                    ),
                    metallic_factor: mtl.pbr_metallic_roughness().metallic_factor(),
                    roughness_factor: mtl.pbr_metallic_roughness().roughness_factor(),
                }
            })
            .collect::<Vec<_>>();

        materials.iter().for_each(|m| {
            info!(
                "Mtl {}, base color {}, metal + rough {}, normals {}",
                m.name, m.base_color_src, m.metallic_src, m.normal_src
            );
        });

        let mut base_color_images = materials
            .iter()
            .map(|mtl| mtl.base_color_src)
            .collect::<Vec<_>>();

        base_color_images.sort();
        base_color_images.dedup();

        self.pixels_base_color = base_color_images
            .iter()
            .enumerate()
            .map(|(idx, base_color)| (idx as u32, *base_color))
            .collect::<Vec<_>>();

        let mut metallic_rough_images = materials
            .iter()
            .map(|mtl| mtl.metallic_src)
            .collect::<Vec<_>>();
        metallic_rough_images.sort();
        metallic_rough_images.dedup();

        self.pixels_metallic_roughness = metallic_rough_images
            .iter()
            .enumerate()
            .map(|(idx, metallic)| (idx as u32, *metallic))
            .collect::<Vec<_>>();

        let mut normal_images = materials
            .iter()
            .map(|mtl| mtl.normal_src)
            .collect::<Vec<_>>();
        normal_images.sort();
        normal_images.dedup();

        self.pixels_normal = normal_images
            .iter()
            .enumerate()
            .map(|(idx, normal)| (idx as u32, *normal))
            .collect::<Vec<_>>();

        let mut pbr_mat_2_gpu_buf = Vec::<PbrMaterial>::with_capacity(gltf_doc.materials().len());

        materials.iter().for_each(|mtl| {
            assert!(self.gltf_mat_2_pbr_mat_mapping.get(&mtl.name).is_none());

            let tex_arr_id_base_color = self
                .pixels_base_color
                .iter()
                .find(|(tex_arr_idx, src_img_idx)| *src_img_idx == mtl.base_color_src)
                .expect("Mapping GLTF material -> PBR material for base color is missing");

            let tex_arr_id_metal_roughness = self
                .pixels_metallic_roughness
                .iter()
                .find(|(tex_arr_idx, src_img_idx)| *src_img_idx == mtl.metallic_src)
                .expect("Mapping GLTF material -> PBR material for metallic+roughness missing");

            let tex_arr_id_normals = self
                .pixels_normal
                .iter()
                .find(|(tex_arr_idx, src_img_idx)| *src_img_idx == mtl.normal_src)
                .expect("Mapping GLTF material -> PBR material for normals missing");

            let pbr_mat_idx = pbr_mat_2_gpu_buf.len() as u32;

            pbr_mat_2_gpu_buf.push(PbrMaterial {
                base_color_factor: mtl.base_color_factor,
                metallic_factor: mtl.metallic_factor,
                roughness_factor: mtl.roughness_factor,
                base_color_texarray_id: tex_arr_id_base_color.0,
                metallic_rough_texarray_id: tex_arr_id_metal_roughness.0,
                normal_texarray_id: tex_arr_id_normals.0,
            });

            self.gltf_mat_2_pbr_mat_mapping
                .insert(mtl.name.clone(), pbr_mat_idx);
        });

        self.pbr_materials = pbr_mat_2_gpu_buf;
    }

    fn process_nodes(&mut self, gltf_doc: &gltf::Document) {
        for s in gltf_doc.scenes() {
            for (idx, nd) in s.nodes().enumerate() {
                self.process_node(&nd, gltf_doc, None);
            }
        }
    }

    fn process_node(&mut self, node: &gltf::Node, gltf_doc: &gltf::Document, parent: Option<u32>) {
        info!("Node {}", node.name().unwrap_or("unnamed"));

        let node_matrix = match node.transform() {
            Transform::Matrix { matrix } => Mat4::from_column_slice(matrix.flat()),
            Transform::Decomposed {
                translation,
                rotation,
                scale,
            } => {
                let s = Mat4::new_nonuniform_scaling(&Vec3::from_row_slice(&scale));
                let r = glm::quat_to_mat4(&glm::Quat::from_vector(Vec4::from_row_slice(&rotation)));
                let t = Mat4::new_translation(&Vec3::from_row_slice(&translation));

                t * r * s
            }
        };

        let node_id = self.nodes.len() as u32;

        self.nodes.push(GeometryNode {
            parent,
            name: node.name().unwrap_or("unknown").into(),
            transform: node_matrix,
        });

        node.children()
            .for_each(|child_node| self.process_node(&child_node, gltf_doc, Some(node_id)));

        if let Some(mesh) = node.mesh().as_ref() {
            let mut matrix = node_matrix;
            let mut parent = parent;

            while let Some(parent_id) = parent {
                let parent_node = &self.nodes[parent_id as usize];
                matrix = parent_node.transform * matrix;
                parent = parent_node.parent;
            }

            self.nodes[node_id as usize].transform = matrix;

            let normals_matrix = glm::transpose(&glm::inverse(&matrix));

            // info!("Processing {}", mesh.name().unwrap_or(""));

            for primitive in mesh.primitives() {
                let mut first_index = self.indices.len() as u32;
                let mut vertex_start = self.vertices.len();
                let mut index_count = 0u32;
                let mtl_name = primitive
                    .material()
                    .name()
                    .expect("Materials without names are not supported chief ...");

                let material_index =
                    *self
                        .gltf_mat_2_pbr_mat_mapping
                        .get(mtl_name)
                        .expect(&format!(
                            "Fatal error: material {} not found in PBR materials buffer.",
                            mtl_name
                        ));

                let reader = primitive.reader(|buf| Some(&self.buffers[buf.index()]));

                let positions = reader.read_positions().expect(&format!(
                    "Missing positions attribute on mesh {}, primitive {}",
                    mesh.name().unwrap_or(""),
                    primitive.index()
                ));

                self.vertices.extend(positions.map(|vtx_pos| {
                    let transformed_pos =
                        matrix * Vec4::new(vtx_pos[0], vtx_pos[1], vtx_pos[2], 1f32);
                    GeometryVertex {
                        pos: transformed_pos.xyz(),
                        pbr_buf_id: material_index,
                        ..GeometryVertex::default()
                    }
                }));

                reader.read_normals().map(|normals| {
                    for (idx, normal) in normals.enumerate() {
                        let n = Vec3::from_column_slice(&normal);
                        let n = glm::normalize(&(matrix * Vec4::new(n.x, n.y, n.z, 0f32)).xyz());
                        self.vertices[vertex_start + idx].normal = n;
                    }
                });

                reader.read_tex_coords(0).map(|texcoords| {
                    for (idx, uv) in texcoords.into_f32().enumerate() {
                        self.vertices[vertex_start + idx].uv = Vec2::from_column_slice(&uv);
                    }
                });

                reader.read_tangents().map(|tangents| {
                    for (idx, tangent) in tangents.enumerate() {
                        self.vertices[vertex_start + idx].tangent =
                            Vec4::from_column_slice(&tangent);
                    }
                });

                reader.read_colors(0).map(|colors| {
                    for (idx, color) in colors.into_rgba_f32().enumerate() {
                        self.vertices[vertex_start + idx].color = Vec4::from_column_slice(&color);
                    }
                });

                self.indices.extend(
                    reader
                        .read_indices()
                        .expect(&format!(
                            "Missing indices on mesh {}, primitive {}",
                            mesh.name().unwrap_or(""),
                            primitive.index(),
                        ))
                        .into_u32()
                        .map(|idx| idx + vertex_start as u32),
                );
            }
        }
    }

    pub fn import_from_file<P: AsRef<std::path::Path>>(file_path: &P) -> Option<ImportedGeometry> {
        let file = std::fs::File::open(file_path.as_ref()).expect(&format!(
            "Failed to open geometry file {}",
            file_path.as_ref().to_str().unwrap()
        ));

        let metadata = file.metadata().expect("Failed to query file metadata!");
        let mapped_file = unsafe {
            MmapOptions::new(metadata.len() as usize)
                .with_file(file, 0)
                .map()
                .expect("Failed to memory map file")
        };

        let (gltf_doc, buffers, images) = gltf::import_slice(mapped_file.as_slice())
            .map_err(|e| error!("GLTF import error: {}", e))
            .ok()?;

        //
        // need RGBA8 for Vulkan
        let images = images
            .into_par_iter()
            .map(|img| match img.format {
                image::Format::R8G8B8 => {
                    let dst = ::image::DynamicImage::ImageRgb8(
                        ::image::RgbImage::from_vec(img.width, img.height, img.pixels)
                            .expect("Error loading GLTF image pixels into RgbImage"),
                    )
                    .into_rgba8();

                    image::Data {
                        pixels: dst.into_vec(),
                        format: image::Format::R8G8B8A8,
                        ..img
                    }
                }

                image::Format::R8G8B8A8 => img,

                _ => img,
            })
            .collect::<Vec<_>>();

        let mut imported = ImportedGeometry {
            nodes: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            buffers,
            images,
            gltf_mat_2_pbr_mat_mapping: HashMap::new(),
            pbr_materials: Vec::new(),
            pixels_base_color: Vec::new(),
            pixels_metallic_roughness: Vec::new(),
            pixels_normal: Vec::new(),
        };

        imported.process_materials(&gltf_doc);
        imported.process_nodes(&gltf_doc);

        Some(imported)
    }
}
