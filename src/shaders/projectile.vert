#version 460 core

layout (location = 0) out VS_OUT_GS_IN {
	flat uint vertex_index;
} vs_out;

void main() {
	vs_out.vertex_index  = gl_VertexIndex;
}
