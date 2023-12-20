#version 460 core

#include "bindless.common.glsl"

layout (location = 0) in vec2 vs_in_pos;
layout (location = 1) in vec2 vs_in_uv;
layout (location = 2) in vec4 vs_in_color;

layout (location = 0) out VS_OUT_FS_IN {
  vec2 uv;
  vec4 color;
  flat uint atlas;
} vs_out;

void main() {
  const UiBackendData ui = g_GlobalUiData[nonuniformEXT(g_GlobalPushConst.id)].arr[0];
  
  gl_Position = ui.ortho * vec4(vs_in_pos, 0.0, 1.0);
  vs_out.uv = vs_in_uv;
  vs_out.color = vs_in_color;
  vs_out.atlas = ui.atlas;
}
