#version 460 core

#extension GL_EXT_nonuniform_qualifier : require

layout (location = 0) in vs_out_fs_in {
  vec3 pos;
  vec3 normal;
  vec2 uv;
  vec4 color;
  vec4 tangent;
  flat uint primitive_id;
  flat uint instance_id;
  flat uint mtl_offset;
} fs_in;

#include "pbr.layout.frag.h"

layout (location = 0) out vec4 FinalFragColor;

void main() {
  const pbr_data_t pbr = primitives_pbr_data.entry[fs_in.mtl_offset];

  vec3 base_color = pbr.base_color_factor.rgb * texture(g_s_colormaps[pbr.colormap_id], fs_in.uv).rgb;
  // vec3 albedo     = pow(base_color, vec3(2.2));
  FinalFragColor = vec4(base_color, 1.0);
}
