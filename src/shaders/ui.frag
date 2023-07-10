#version 460 core

#extension GL_EXT_nonuniform_qualifier : require

layout (location = 0) in VS_OUT_FS_IN {
  vec2 uv;
  vec4 color;
} fs_in;

#include "pbr.layout.frag.h"

layout (location = 0) out vec4 FinalFragColor;

void main() {
  FinalFragColor = vec4(texture(s_misc_textures[g_misc_data.atlas_id], fs_in.uv).r) * fs_in.color;
}
