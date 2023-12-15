// for nonuniformEXT
#extension GL_EXT_nonuniform_qualifier : require
// for gl_DrawID
#extension GL_ARB_shader_draw_parameters : require

#if defined(BINDLESS_VS_SETUP)

struct GlobalData {
  mat4 matrices[4];
  vec4 vectors[16];
  uint integers[64];
};

layout (set = 0, binding = 0) readonly buffer BufferGlobalData {
  GlobalData data[];
} g_BufferGlobal;

struct InstanceRenderInfo {
  mat4 model;
  uint mtl_coll_offset;
};

layout (std430, set = 1, binding = 0) readonly buffer InstancedData_t {
  InstanceRenderInfo data[]; 
} instances;

#endif

#if defined(BINDLESS_FS_SETUP)

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

struct GlobalData {
  mat4 matrices[4];
  vec4 vectors[16];
  uint integers[64];
};

layout (set = 12, binding = 0) readonly buffer BufferGlobalData {
  GlobalData data[];
} g_BufferGlobal;

#endif
