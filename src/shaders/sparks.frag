#version 460 core

layout (set = 0, binding = 1) uniform sampler2D s;

layout (location = 0) in VS_OUT_FS_IN {
	flat vec3 color;
} fs_in;

layout (location = 0) out vec4 FinalFragColor;

void main() {
	FinalFragColor = vec4(fs_in.color, 1.0) * texture(s, gl_PointCoord);
}
