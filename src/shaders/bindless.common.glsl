// for nonuniformEXT
#extension GL_EXT_nonuniform_qualifier : require
// for gl_DrawID
#extension GL_ARB_shader_draw_parameters : require

struct GlobalUniformData {
  mat4 projection_view;
  mat4 view;
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

layout (set = 1, binding = 0) readonly buffer GlobalUiData {
  UiBackendData arr[];
} g_GlobalUiData[];

layout (set = 1, binding = 0) readonly buffer GlobalSkyboxData {
  SkyboxData arr[];
} g_GlobalSkyboxData[];

layout (set = 2, binding = 0) uniform sampler2D g_Global2DTextures[];
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