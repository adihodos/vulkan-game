#version 460 core

#include "bindless.common.glsl"

layout (location = 0) out VS_OUT_FS_IN {
  vec4 color;
} vs_out;

void main() {
  const uint uboHandle = g_GlobalPushConst.id >> 16;
  const uint sparkHandle = g_GlobalPushConst.id & 0xFFFF;

  const SparkInstance si = g_GlobalSparkInstances[nonuniformEXT(sparkHandle)].arr[gl_InstanceIndex * 2 + gl_VertexIndex];
  const mat4 pvMatrix = g_GlobalUniform[nonuniformEXT(uboHandle)].data[0].projection_view;

  gl_Position = pvMatrix * vec4(si.pos, 1.0);
  vs_out.color = vec4(si.color, si.intensity);
}
