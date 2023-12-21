#version 460 core

#include "bindless.common.glsl"

layout (location = 0) in vec2 vs_in_pos;
layout (location = 1) in vec2 vs_in_uv;
layout (location = 2) in uint vs_in_texid;
layout (location = 3) in vec4 vs_in_color;


layout (location = 0) out VS_OUT_FS_IN {
  vec4 color;
  vec2 uv;
  flat uint texid;
} vs_out;

void main() {
  const uint idxGlobalUniform = g_GlobalPushConst.id & 0xFFFF;
  const mat4 transformMatrix = g_GlobalUniform[nonuniformEXT(idxGlobalUniform)].data[0].orthographic;
  gl_Position =  transformMatrix * vec4(vs_in_pos, 0.0, 1.0);
  vs_out.color = vs_in_color;
  vs_out.uv = vs_in_uv;
  vs_out.texid = vs_in_texid;
}

