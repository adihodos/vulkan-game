#version 460 core

layout (location = 0) in vs_out_fs_in {
	vec3 pos;
	vec3 normal;
	vec2 uv;
	vec4 color;
	vec4 tangent;
	flat uint primitive_id;
} fs_in;

layout (set = 1, binding = 0) uniform sampler2D s_colormap;
layout (location = 0) out vec4 FinalFragColor;

void main() {
  vec3 base_color = texture(s_colormap, fs_in.uv).rgb;
  vec3 albedo     = pow(base_color, vec3(2.2));
  FinalFragColor = vec4(albedo, 1.0);
}
