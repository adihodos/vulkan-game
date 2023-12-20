#version 460 core

#include "bindless.common.glsl"

layout (location = 0) in VS_OUT_FS_IN {
  vec2 uv;
  vec4 color;
  flat uint atlas;
} fs_in;

layout (location = 0) out vec4 FinalFragColor;

void main() {
  FinalFragColor = texture(g_Global2DTextures[fs_in.atlas], fs_in.uv).r * fs_in.color;
}
