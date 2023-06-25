#version 460 core

layout (push_constant) uniform transforms_t {
	mat4 view_matrix;
} transforms;

layout (location = 0) out VS_OUT_FS_IN {
	vec3 texcoords;
} vs_out;

const vec2 QUAD_VERTICES[] = vec2[](
	vec2(-1.0, 1.0),
	vec2(-1.0, -1.0),
	vec2(1.0, -1.0),
	vec2(1.0, 1.0)
);

void main() {
	vs_out.texcoords = mat3(transforms.view_matrix) * vec3(QUAD_VERTICES[gl_VertexIndex], 1.0);
	vs_out.texcoords.y *= +1.0;
	gl_Position = vec4(QUAD_VERTICES[gl_VertexIndex], 1.0, 1.0);
}
