#version 460 core

#extension GL_EXT_nonuniform_qualifier : require

#include "pbr.layout.frag.h"

layout (location = 0) in VS_OUT_FS_IN {
  vec3 texcoords;
} fs_in;

layout (location = 0) out vec4 FinalFragColor;

void main() {
  FinalFragColor = vec4(texture(s_prefiltered[lighting_data.skybox], fs_in.texcoords).rgb, 1.0);
}
