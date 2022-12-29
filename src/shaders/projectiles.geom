#version 460 core

layout (points) in;
layout (triangle_strip, max_vertices = 24) out;

struct InstanceData {
	mat4 model;
};

layout (set = 0, binding = 0) uniform Transforms {
	mat4 view_projection;
	vec3 half_extent;
} transforms;

layout (std140, set = 0, binding = 1) readonly buffer Instances {
	InstanceData data[];
} instances;

layout (location = 0) in VS_OUT_GS_IN {
	flat uint vertex_index;
} gs_in[];

layout (location = 0) out GS_OUT_FS_IN {
	vec3 color;
} gs_out;

const vec3 ORANGE = vec3(1.0, 0.541, 0.0);
const vec3 ORANGE_LIGHT = vec3(1.0, 0.831, 0.635);

struct VertexPC {
	vec3 pos;
	vec3 color;
};

const VertexPC VERTICES_YZ[] = VertexPC[] (
	VertexPC(vec3(0.0, 0.5, 0.0), ORANGE_LIGHT),
	VertexPC(vec3(0.0, 0.5, 0.5), ORANGE),
	VertexPC(vec3(0.0, -0.5, 0.5), ORANGE),
	VertexPC(vec3(0.0, -0.5, 0.0), ORANGE_LIGHT),

	VertexPC(vec3(0.0, 0.5, 0.0), ORANGE_LIGHT),
	VertexPC(vec3(0.0, 0.5, -0.5), ORANGE),
	VertexPC(vec3(0.0, -0.5, -0.5), ORANGE),
	VertexPC(vec3(0.0, -0.5, 0.0), ORANGE_LIGHT)
);

const VertexPC VERTICES_XZ[] = VertexPC[] (
	VertexPC(vec3(-0.5, 0.0, 0.0), ORANGE_LIGHT),
	VertexPC(vec3(-0.5, 0.0, 0.5), ORANGE),
	VertexPC(vec3(0.5, 0.0, 0.5), ORANGE),
	VertexPC(vec3(0.5, 0.0, 0.0), ORANGE_LIGHT),

	VertexPC(vec3(-0.5, 0.0, 0.0), ORANGE_LIGHT),
	VertexPC(vec3(-0.5, 0.0, -0.5), ORANGE),
	VertexPC(vec3(0.5, 0.0, -0.5), ORANGE),
	VertexPC(vec3(0.5, 0.0, 0.0), ORANGE_LIGHT)
);

const uint indices[] = uint[](0, 1, 2, 0, 2, 3);

void gen_quads(in VertexPC src_vertices[8]) {
	//
	//
	for (uint i = 0; i < 2; ++i) {
		for (uint j = 0; j < 2; ++j) {
			VertexPC v0 = src_vertices[i * 4 + indices[j * 3 + 0]];
			vec4 pos = vec4(v0.pos * transforms.half_extent, 1.0);
			gl_Position = transforms.view_projection * instances.data[gs_in[0].vertex_index].model * pos;
			gs_out.color = v0.color;
			EmitVertex();

			VertexPC v1 = src_vertices[i * 4 + indices[j * 3 + 1]];
			pos = vec4(v1.pos * transforms.half_extent, 1.0);
			gl_Position = transforms.view_projection * instances.data[gs_in[0].vertex_index].model * pos;
			gs_out.color = v1.color;
			EmitVertex();

			VertexPC v2 = src_vertices[i * 4 + indices[j * 3 + 2]];
			pos = vec4(v2.pos * transforms.half_extent, 1.0);
			gl_Position = transforms.view_projection * instances.data[gs_in[0].vertex_index].model * pos;
			gs_out.color = v2.color;
			EmitVertex();

			EndPrimitive();
		}
	}
}

void main() {
	gen_quads(VERTICES_YZ);
	gen_quads(VERTICES_XZ);
}
