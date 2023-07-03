#version 460 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 uv;

layout (set = 0, binding = 0) uniform ProjGlobals {
  mat4 proj_view;
} prj_globals;

struct ProjectileInstance {
  mat4 model2world;
  vec4 inner_color;
  vec4 outer_color;
};

layout (set = 0, binding = 1) readonly buffer ProjInstanceData {
  ProjectileInstance data[];
} instances;

layout (location = 0) out VS_OUT_GS_IN {
  float mix_factor;
  vec4 inner_color;
  vec4 outer_color;
  vec2 uv;
} vs_out;

void main() {
  const ProjectileInstance inst = instances.data[gl_InstanceIndex];
  gl_Position = prj_globals.proj_view * inst.model2world * vec4(pos, 1.0);

  vs_out.mix_factor = 0.75;
  vs_out.inner_color = inst.inner_color;
  vs_out.outer_color = inst.outer_color;
  vs_out.uv = uv;
}
