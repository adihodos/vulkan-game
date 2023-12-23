#version 460 core

#include "bindless.common.glsl"

layout (location = 0) in VS_OUT_FS_IN {
  vec3 texcoords;
} fs_in;

layout (location = 0) out vec4 FinalFragColor;

void main() {
  const SkyboxData skyboxData = g_GlobalSkyboxData[nonuniformEXT(g_GlobalPushConst.id)].arr[0];
  FinalFragColor = vec4(texture(g_GlobalCubeTextures[nonuniformEXT(skyboxData.skyboxPrefiltered)], fs_in.texcoords).rgb, 1.0);
}
