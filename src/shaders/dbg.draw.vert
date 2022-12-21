#version 460 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 endpos;
layout (location = 2) in vec4 color_start;
layout (location = 3) in vec4 color_end;

layout (location = 0) out VS_OUT_GS_IN {
    vec3 pos;
    vec3 endpos;
    vec4 color_start;
    vec4 color_end;
} vs_out;

void main() {
    vs_out.pos = pos;
    vs_out.endpos = endpos;
    vs_out.color_start = color_start;
    vs_out.color_end = color_end;
}

