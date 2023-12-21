#version 460 core

#include "bindless.common.glsl"

layout (points) in;
layout (line_strip, max_vertices = 2) out;

layout (location = 0) in VS_OUT_GS_IN {
  vec3 pos;
  vec3 endpos;
  vec4 color_start;
  vec4 color_end;
} gs_in[];

layout (location = 0) out GS_OUT_FS_IN {
  vec3 color;
} gs_out;

void main() {
  const mat4 worldViewProj = g_GlobalUniform[nonuniformEXT(g_GlobalPushConst.id)].data[0].projection_view;
  
  gl_Position = worldViewProj * vec4(gs_in[0].pos, 1.0);
  gs_out.color = gs_in[0].color_start.rgb;
  EmitVertex();

  gl_Position = worldViewProj * vec4(gs_in[0].endpos, 1.0);
  gs_out.color = gs_in[0].color_end.rgb;
  EmitVertex();

  EndPrimitive();
}
