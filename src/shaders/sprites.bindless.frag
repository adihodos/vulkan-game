#version 460 core

#include "bindless.common.glsl"

layout (location = 0) in VS_OUT_FS_IN {
  vec4 color;
  vec2 uv;
  flat uint texid;
} fs_in;

layout (location = 0) out vec4 FinalFragColor;

void main() {
  const uint spriteTexIdx = g_GlobalPushConst.id >> 16;
  FinalFragColor =
    fs_in.color *
    texture(g_Global2DArrayTextures[nonuniformEXT(spriteTexIdx)],
	    vec3(fs_in.uv, fs_in.texid)).rrrg;
}
