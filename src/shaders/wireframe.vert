#version 460 core

layout (location = 0) in vec3 position;

layout (std140, set = 0, binding = 0) uniform Transforms {
	mat4 world_view_proj;
	vec4 color;
} transforms;

layout (location = 0) out VS_OUT_FS_IN {
	vec4 color;
} vs_out;

void main() {
	gl_Position = transforms.world_view_proj * vec4(position, 1.0);
	vs_out.color = transforms.color;
}
