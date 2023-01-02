#version 460 core

layout (set = 1, binding = 0) uniform sampler2DArray s;

layout (location = 0) in VS_OUT_FS_IN {
	vec4 color;
	vec2 uv;
	flat uint texid;
} fs_in;

layout (location = 0) out vec4 FinalFragColor;

void main() {
	FinalFragColor = fs_in.color * texture(s, vec3(fs_in.uv, fs_in.texid));
}
