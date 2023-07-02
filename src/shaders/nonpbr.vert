#version 460 core

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
} vs_out;

layout (std140, set = 0, binding = 0) uniform transforms_t {
	mat4 projection;
	mat4 view;
#if !defined(INSTANCED_RENDERING)
	mat4 model;
#endif	
} transforms;

#if defined(INSTANCED_RENDERING)
layout (std140, set = 0, binding = 1) readonly buffer instanced_data_t {
	mat4 model[];
} instances;
#endif

void main() {
	mat4 model = 
#if defined(INSTANCED_RENDERING)
		instances.model[gl_InstanceIndex];
#else
		transforms.model;
#endif

	vs_out.pos = (model * vec4(vs_in_pos, 1.0)).xyz;
	vs_out.normal = mat3(model) * vs_in_normal;
	vs_out.uv = vs_in_uv;
	vs_out.color = vs_in_color;
	vs_out.tangent = vs_in_tangent;
	vs_out.primitive_id = vs_in_primitive_id;

	gl_Position = transforms.projection * transforms.view * vec4(vs_out.pos, 1.0);
}
