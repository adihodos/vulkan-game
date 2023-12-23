#ifndef PBR_LAYOUT_VERT_H_INCLUDED
#define PBR_LAYOUT_VERT_H_INCLUDED

struct InstanceRenderInfo {
  mat4 model;
  uint mtl_coll_offset;
};

layout (std140, set = 0, binding = 0) uniform GlobalData_t {
  mat4 projection_view;
  mat4 view;
} g_data;

layout (std430, set = 1, binding = 0) readonly buffer InstancedData_t {
  InstanceRenderInfo data[]; 
} instances;

layout (push_constant) uniform GlobalMiscData_t {
  mat4 model;
  uint atlas_id;
} g_misc_data;

#endif // PBR_LAYOUT_VERT_H_INCLUDED
