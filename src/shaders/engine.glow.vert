#version 460 core

#include "bindless.common.glsl"

layout (location = 0) in vec3 vs_in_pos;
layout (location = 1) in vec3 vs_in_normal;
layout (location = 2) in vec2 vs_in_uv;
layout (location = 3) in vec4 vs_in_color;
layout (location = 4) in vec4 vs_in_tangent;
layout (location = 5) in uint vs_in_primitive_id;

layout (location = 0) out vs_out_fs_in {
  vec3 pos;
  vec2 uvGlow;
  vec2 uvNoise;
  vec3 color;
  flat uint glowImage;
  flat uint noiseImage;
  flat float glowIntensity;
} vs_out;

void main() {
  const uint idx = g_GlobalPushConst.id;
  const EngineGlowData e = g_GlobalEngineGlowData[nonuniformEXT(idx)].arr[0];
  const InstanceRenderInfo ri = g_GlobalInstances[nonuniformEXT(e.instanceHandle)].data[gl_InstanceIndex];

  mat4 model = ri.model;

  vs_out.pos = (model * vec4(vs_in_pos, 1.0)).xyz;
  vs_out.uvGlow = vs_in_uv;
  vs_out.uvNoise = (mat3(e.texTransform) * vec3(vs_in_uv, 1.0)).xy;
  vs_out.color = e.glowColor;
  vs_out.glowImage = e.glowImage;
  vs_out.noiseImage = e.noiseImage;
  vs_out.glowIntensity = e.glowIntensity;

  gl_Position = g_GlobalUniform[nonuniformEXT(e.uboHandle)].data[0].projection_view * vec4(vs_out.pos, 1.0);
}
