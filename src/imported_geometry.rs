use std::{collections::BinaryHeap, mem::size_of};

use gltf::{
    buffer::{self, Data},
    image,
    mesh::Mode,
    scene::Transform,
    Document, Node,
};
use log::{error, info};
use mmap_rs::MmapOptions;
use nalgebra_glm::{identity, quat, translate, Mat4, Qua, Vec2, Vec3, Vec4};
use slice_of_array::prelude::*;

use nalgebra_glm as glm;

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
    pub data: u32,
}

impl std::default::Default for GeometryVertex {
    fn default() -> Self {
        GeometryVertex {
            pos: Vec3::new(0f32, 0f32, 0f32),
            normal: Vec3::new(0f32, 0f32, 0f32),
            uv: Vec2::new(0f32, 0f32),
            color: Vec4::new(0f32, 0f32, 0f32, 1f32),
            tangent: Vec4::new(0f32, 0f32, 0f32, 0f32),
            data: 0u32, // ..Default::default()
        }
    }
}

pub struct ImportedGeometry {
    nodes: Vec<GeometryNode>,
    vertices: Vec<GeometryVertex>,
    indices: Vec<u32>,
}

impl ImportedGeometry {
    fn new() -> ImportedGeometry {
        ImportedGeometry {
            nodes: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
        }
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

    fn process_node(
        &mut self,
        node: &gltf::Node,
        buffers: &[buffer::Data],
        images: &[image::Data],
        parent: Option<u32>,
    ) {
        info!("Node {}", node.name().unwrap_or("unnamed"));

        // node.transform().decomposed()

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
            .for_each(|child_node| self.process_node(&child_node, buffers, images, Some(node_id)));

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

            info!("Processing {}", mesh.name().unwrap_or(""));

            for primitive in mesh.primitives() {
                let mut first_index = self.indices.len() as u32;
                let mut vertex_start = self.vertices.len();
                let mut index_count = 0u32;
                let mut material_index = 0u32;

                info!(
                    "Material {}",
                    primitive
                        .material()
                        .name()
                        .expect("Primitives without materials are not supported")
                );

                let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

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

        let mut geometry = ImportedGeometry::new();

        info!(
            "Nodes {}, materials {}",
            gltf_doc.nodes().len(),
            gltf_doc.materials().len()
        );

        let materials = gltf_doc
            .materials()
            .map(|mtl| mtl.name().expect("Unnamed materials not supported"))
            .collect::<BinaryHeap<_>>();

        for mtl in gltf_doc.materials() {
            // mtl.pbr_metallic_roughness().metallic_roughness_texture().
        }

        // materials.

        // materials
        //     .iter()
        //     .enumerate()
        //     .for_each(|(idx, mtl)| info!("Material {} -> {}", idx, mtl));

        // for mtl in gltf_doc.materials() {
        //     // mtl.index()
        // }
        // geometry.nodes.extend(
        //     gltf_doc
        //         .nodes()
        //         .enumerate()
        //         .map(|(node_idx, node)| GeometryNode {
        //             idx: node_idx as u32,
        //             ..GeometryNode::default()
        //         }),
        // );

        for s in gltf_doc.scenes() {
            for (idx, nd) in s.nodes().enumerate() {
                geometry.process_node(&nd, &buffers, &images, None)
            }
        }

        // for node in gltf_doc.nodes() {
        //     // info!("processing node {}", node.name().unwrap_or(""));
        //     geometry.process_node(&node, &gltf_doc, &buffers, &images, None);
        // }

        Some(geometry)
    }
}
