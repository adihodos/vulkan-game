// for nonuniformEXT
#extension GL_EXT_nonuniform_qualifier : require
// for gl_DrawID
#extension GL_ARB_shader_draw_parameters : require

struct GlobalUniformData {
  mat4 projection_view;
  mat4 view;
  mat4 orthographic;
  vec3 eyePosition;
  uint frameId;
};

layout (std140, set = 0, binding = 0) uniform GlobalData_t {
  GlobalUniformData data[];
} g_GlobalUniform[];

struct UiBackendData {
  mat4 ortho;
  uint atlas;
};

struct SkyboxData {
  uint globalUboHandle;
  uint skyboxPrefiltered;
  uint skyboxIrradiance;
  uint skyboxBRDFLut;
};

struct InstanceRenderInfo {
  mat4 model;
  uint mtl_coll_offset;
};

struct PbrData {
  vec4 base_color_factor;
  float metallic_factor;
  float roughness_factor;
  uint colormap_id;
  uint metallic_roughness_id;
  uint normal_id;
};

struct PbrRenderpassHandles {
  uint uboHandle;
  uint instHandle;
  uint pbrMtlHandle;
  uint skyboxHandle;
};

layout (set = 1, binding = 0) readonly buffer GlobalInstanceData {
  InstanceRenderInfo data[]; 
} g_GlobalInstances[];

layout (set = 1, binding = 0) readonly buffer GlobalUiData {
  UiBackendData arr[];
} g_GlobalUiData[];

layout (set = 1, binding = 0) readonly buffer GlobalSkyboxData {
  SkyboxData arr[];
} g_GlobalSkyboxData[];

layout (set = 1, binding = 0) readonly buffer GlobalPbrData {
  PbrData arr[];
} g_GlobalPbrData[];

layout (set = 1, binding = 0) readonly buffer GlobalPbrRenderpassHandles {
  PbrRenderpassHandles arr[];
} g_GlobalPbrHandles[];

layout (set = 2, binding = 0) uniform sampler2D g_Global2DTextures[];
layout (set = 2, binding = 0) uniform sampler2DArray g_Global2DArrayTextures[];
layout (set = 2, binding = 0) uniform samplerCube g_GlobalCubeTextures[];

layout (push_constant) uniform GlobalPushConstant {
  uint id;
} g_GlobalPushConst;  

// unpack a frameid and resource handle
// return uvec2{resource, frame}
uvec2 unpack_frame_and_resource_data() {
  const uint id = g_GlobalPushConst.id;
  return uvec2(id >> 4, id & 0xF);
}
