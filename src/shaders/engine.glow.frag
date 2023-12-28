#version 460 core

#include "bindless.common.glsl"

layout (location = 0) in vs_out_fs_in {
  vec3 pos;
  vec2 uvGlow;
  vec2 uvNoise;
  vec3 color;
  flat uint glowImage;
  flat uint noiseImage;
  flat float glowIntensity;
} fs_in;

layout (location = 0) out vec4 FinalFragColor;

void main() {
  const float a = texture(g_Global2DTextures[nonuniformEXT(fs_in.glowImage)], fs_in.uvGlow).r;
  
  FinalFragColor = vec4(
    fs_in.color
    *
    fs_in.glowIntensity
    *
    texture(g_Global2DTextures[nonuniformEXT(fs_in.noiseImage)], fs_in.uvNoise).rgb,
    a);
}
