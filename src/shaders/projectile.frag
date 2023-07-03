#version 460 core

layout (location = 0) out vec4 FinalFragColor;

layout (location = 0) in VS_OUT_GS_IN {
  float mix_factor;
  vec4 inner_color;
  vec4 outer_color;
  vec2 uv;
} fs_in;

void main() {
  FinalFragColor = fs_in.outer_color;
}
