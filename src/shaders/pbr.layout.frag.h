#ifndef PBR_LAYOUT_FRAG_H_INCLUDED
#define PBR_LAYOUT_FRAG_H_INCLUDED

struct pbr_data_t {
  vec4 base_color_factor;
  float metallic_factor;
  float roughness_factor;
  uint colormap_id;
  uint metallic_roughness_id;
  uint normal_id;
};

layout (std140, set = 2, binding = 0) readonly buffer primitive_pbr_data_t {
  pbr_data_t entry[];
} primitives_pbr_data;

layout (set = 3, binding = 0) uniform sampler2D g_s_colormaps[];
layout (set = 4, binding = 0) uniform sampler2D g_s_metal_roughness_maps[];
layout (set = 5, binding = 0) uniform sampler2D g_s_normal_maps[];

layout (std140, set = 6, binding = 0) uniform pbr_lighting_data_t {
  vec3 eye_pos;
  uint skybox;
} lighting_data;

layout (set = 7, binding = 0) uniform samplerCube s_irradiance[];
layout (set = 8, binding = 0) uniform samplerCube s_prefiltered[];
layout (set = 9, binding = 0) uniform sampler2D s_brdf_lut[];

layout (set = 10, binding = 0) uniform sampler2D s_misc_textures[];
layout (set = 11, binding = 0) uniform sampler2DArray s_misc_arr_textures[];

/* layout (push_constant) uniform MiscData { */
/*   uint misc_tex_id; */
/*   uint misc_arr_tex_id; */
/* } g_misc_data; */

layout (push_constant) uniform GlobalMiscData_t {
  mat4 model;
  uint atlas_id;
} g_misc_data;

#endif // PBR_LAYOUT_FRAG_H_INCLUDED
