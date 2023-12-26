use crate::math::AABB3;
use crate::resource_system::PbrMaterial;
use gltf::{buffer, image, scene::Transform};
use itertools::Itertools;
use mmap_rs::MmapOptions;
use nalgebra_glm as glm;
use nalgebra_glm::{Mat4, Vec2, Vec3, Vec4};
use rayon::prelude::*;
use slice_of_array::prelude::*;

#[derive(Clone, Debug)]
pub struct GeometryNode {
    pub parent: Option<u32>,
    pub name: String,
    pub transform: Mat4,
    pub aabb: AABB3,
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub index_count: u32,
}

impl std::default::Default for GeometryNode {
    fn default() -> Self {
        GeometryNode {
            parent: None,
            name: String::new(),
            transform: glm::identity::<f32, 4>(),
            aabb: AABB3::identity(),
            vertex_offset: 0,
            index_offset: 0,
            index_count: 0,
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
            pbr_buf_id: 0u32,
        }
    }
}

pub struct ImportedGeometry {
    nodes: Vec<GeometryNode>,
    vertices: Vec<GeometryVertex>,
    indices: Vec<u32>,
    buffers: Vec<buffer::Data>,
    images: Vec<image::Data>,
    pub materials: Vec<(String, PbrMaterial)>,
    pub aabb: AABB3,
}

impl ImportedGeometry {
    pub fn get_image_data(&self, image_index: usize) -> &image::Data {
        &self.images[image_index]
    }

    pub fn nodes(&self) -> &[GeometryNode] {
        &self.nodes
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
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

    fn process_materials(&mut self, gltf: &gltf::Document) {
        self.materials = gltf
            .materials()
            .filter_map(|mtl| {
                let name = mtl
                    .name()
                    .expect("Unnamed materials are not supported!")
                    .to_string();

                if mtl.pbr_metallic_roughness().base_color_texture().is_none() {
                    return None;
                }

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

                Some((
                    name,
                    PbrMaterial {
                        base_color_factor: Vec4::from_row_slice(
                            &mtl.pbr_metallic_roughness().base_color_factor(),
                        ),
                        metallic_factor: mtl.pbr_metallic_roughness().metallic_factor(),
                        roughness_factor: mtl.pbr_metallic_roughness().roughness_factor(),
                        base_color_texarray_id: base_color_src,
                        metallic_rough_texarray_id: metalic_roughness_src,
                        normal_texarray_id: normal_src,
                    },
                ))
            })
            .collect();

        self.materials.sort_by_key(|(name, _)| name.clone());
    }

    fn process_nodes(&mut self, gltf_doc: &gltf::Document) {
        for s in gltf_doc.scenes() {
            for (_idx, nd) in s.nodes().enumerate() {
                self.process_node(&nd, gltf_doc, None);
            }
        }
    }

    fn process_node(&mut self, node: &gltf::Node, gltf_doc: &gltf::Document, parent: Option<u32>) {
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
            aabb: AABB3::identity(),
            vertex_offset: self.vertices.len() as u32,
            index_offset: self.indices.len() as u32,
            index_count: 0,
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

            for primitive in mesh.primitives() {
                let _first_index = self.indices.len() as u32;
                let vertex_start = self.vertices.len();
                let _index_count = 0u32;
                let mtl_name = primitive
                    .material()
                    .name()
                    .expect("Materials without names are not supported chief ...");

                let material_index = self
                    .materials
                    .iter()
                    .find_position(|(name, _)| name == mtl_name)
                    .map(|(pos, _)| pos as u32)
                    .unwrap_or(0);

                let reader = primitive.reader(|buf| Some(&self.buffers[buf.index()]));

                let positions = reader.read_positions().expect(&format!(
                    "Missing positions attribute on mesh {}, primitive {}",
                    mesh.name().unwrap_or(""),
                    primitive.index()
                ));

                self.vertices.extend(positions.map(|vtx_pos| {
                    let transformed_pos =
                        matrix * Vec4::new(vtx_pos[0], vtx_pos[1], vtx_pos[2], 1f32);

                    self.nodes[node_id as usize]
                        .aabb
                        .add_point(transformed_pos.xyz());

                    GeometryVertex {
                        pos: transformed_pos.xyz(),
                        pbr_buf_id: material_index,
                        ..GeometryVertex::default()
                    }
                }));

                reader.read_normals().map(|normals| {
                    for (idx, normal) in normals.enumerate() {
                        let n = (normals_matrix * glm::vec4(normal[0], normal[1], normal[2], 0f32))
                            .xyz();
                        let n = glm::normalize(&n);

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

                let indices = self.indices.len();
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
                self.nodes[node_id as usize].index_count += (self.indices.len() - indices) as u32;
            }
        }
    }

    pub fn import_from_file<P: AsRef<std::path::Path>>(file_path: &P) -> Option<ImportedGeometry> {
        let file = std::fs::File::open(file_path.as_ref())
            .map_err(|e| {
                log::error!(
                    "Failed to open file {}, error: {e}",
                    file_path.as_ref().display()
                )
            })
            .ok()?;

        let metadata = file
            .metadata()
            .map_err(|e| {
                log::error!(
                    "Failed to query file {} metadata, error: {e}",
                    file_path.as_ref().display(),
                )
            })
            .ok()?;

        let mapped_file = unsafe {
            MmapOptions::new(metadata.len() as usize)
                .map_err(|e| {
                    log::error!(
                        "Failed to create mapping options for file {}, error: {e}",
                        file_path.as_ref().display()
                    )
                })
                .ok()?
                .with_file(&file, 0)
                .map()
                .map_err(|e| {
                    log::error!(
                        "Failed to mmap file {}, error: {e}",
                        file_path.as_ref().display()
                    )
                })
                .ok()?
        };

        let (gltf_doc, buffers, images) = gltf::import_slice(mapped_file.as_slice())
            .map_err(|e| log::error!("GLTF import error: {}", e))
            .ok()?;

        log::info!("Image count: {}", images.len());

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
            aabb: AABB3::identity(),
            materials: Vec::new(),
        };

        imported.process_materials(&gltf_doc);
        log::info!("Maerials:{:#?}", imported.materials);

        imported.process_nodes(&gltf_doc);
        imported.compute_aabb();

        // log::info!(
        //     "imported.gltf_mat_2_pbr_mat_mapping : {:?}",
        //     imported.gltf_mat_2_pbr_mat_mapping
        // );
        // log::info!("PBR materials: {:#?}", imported.pbr_materials);

        Some(imported)
    }

    fn compute_aabb(&mut self) {
        self.aabb = self
            .nodes
            .iter()
            .fold(AABB3::identity(), |aabb, current_node| {
                crate::math::aabb_merge(&aabb, &current_node.aabb)
            });
    }

    pub fn has_materials(&self) -> bool {
        !self.materials.is_empty()
    }
}
