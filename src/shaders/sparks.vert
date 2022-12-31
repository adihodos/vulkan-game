#version 460 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 color;
layout (location = 2) in float intensity;

layout (set = 0, binding = 0) uniform Transforms {
	mat4 view_projection;
} transforms;

layout (location = 0) out VS_OUT_FS_IN {
	flat vec3 color;
} vs_out;

void main() {
	const float size = 4.0;

	vs_out.color = intensity * color;
	gl_Position = transforms.view_projection * vec4(pos, 1.0);
	gl_PointSize = size;
}
