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
  vec3 normal;
  vec2 uv;
  vec4 color;
  vec4 tangent;
  flat uint primitive_id;
  flat uint instance_id;
  flat uint mtl_offset;
} vs_out;

void main() {
  const uint idx = g_GlobalPushConst.id;
  const PbrRenderpassHandles handles = g_GlobalPbrHandles[nonuniformEXT(idx)].arr[0];
  const uint ubo_idx = handles.uboHandle;
  const uint inst_idx = handles.instHandle;
  
  const InstanceRenderInfo inst_info = g_GlobalInstances[nonuniformEXT(inst_idx)].data[gl_InstanceIndex];
  
  mat4 model = inst_info.model;

  vs_out.pos = (model * vec4(vs_in_pos, 1.0)).xyz;
  vs_out.normal = mat3(model) * vs_in_normal;
  vs_out.uv = vs_in_uv;
  vs_out.color = vs_in_color;
  vs_out.tangent = vs_in_tangent;
  vs_out.primitive_id = vs_in_primitive_id;
  vs_out.instance_id = gl_InstanceIndex;
  vs_out.mtl_offset = inst_info.mtl_coll_offset;

  gl_Position = g_GlobalUniform[nonuniformEXT(ubo_idx)].data[0].projection_view * vec4(vs_out.pos, 1.0);
}
