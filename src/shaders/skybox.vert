#version 460 core

#include "pbr.layout.vert.h"

layout (location = 0) out VS_OUT_FS_IN {
  vec3 texcoords;
} vs_out;

// indices = [0, 3, 2, 0, 2, 1];
const vec2 QUAD_VERTICES[] = vec2[](
				    vec2(-1.0, 1.0),
				    vec2(1.0, 1.0),
				    vec2(1.0, -1.0),
				    vec2(-1.0, 1.0),
				    vec2(1.0, -1.0),
				    vec2(-1.0, -1.0)
				    );

void main() {
  vs_out.texcoords = mat3(g_data.view) * vec3(QUAD_VERTICES[gl_VertexIndex], 1.0);
  vs_out.texcoords.y *= +1.0;
  gl_Position = vec4(QUAD_VERTICES[gl_VertexIndex], 1.0, 1.0);
}
