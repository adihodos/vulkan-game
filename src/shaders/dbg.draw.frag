#version 460 core

layout (location = 0) in GS_OUT_FS_IN {
	vec3 color;
} fs_in;

layout (location = 0) out vec4 FinalFragColor;

void main() {
	FinalFragColor = vec4(fs_in.color, 1.0);
}
