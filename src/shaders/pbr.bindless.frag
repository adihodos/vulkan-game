#version 460 core

#extension GL_EXT_nonuniform_qualifier : require

layout (location = 0) in vs_out_fs_in {
  vec3 pos;
  vec3 normal;
  vec2 uv;
  vec4 color;
  vec4 tangent;
  flat uint primitive_id;
  flat uint instance_id;
  flat uint mtl_offset;
} fs_in;

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

layout (location = 0) out vec4 FinalFragColor;

const float PI = 3.14159265359;

//
// PBR code adapted from https://learnopengl.com/PBR/Lighting

// ----------------------------------------------------------------------------
// Easy trick to get tangent-normals to world-space to keep PBR code simplified.
// Don't worry if you don't get what's going on; you generally want to do normal 
// mapping the usual way for performance anways; I do plan make a note of this 
// technique somewhere later in the normal mapping tutorial.
vec3 getNormalFromMap(vec3 normal)
{
    vec3 tangentNormal = normal * 2.0 - 1.0;

    vec3 Q1  = dFdx(fs_in.pos);
    vec3 Q2  = dFdy(fs_in.pos);
    vec2 st1 = dFdx(fs_in.uv);
    vec2 st2 = dFdy(fs_in.uv);

    vec3 N   = normalize(fs_in.normal);
    vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
    vec3 B  = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}
// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
// ----

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Converts a color from linear light gamma to sRGB gamma
vec4 fromLinear(vec4 linearRGB)
{
    bvec3 cutoff = lessThan(linearRGB.rgb, vec3(0.0031308));
    vec3 higher = vec3(1.055)*pow(linearRGB.rgb, vec3(1.0/2.4)) - vec3(0.055);
    vec3 lower = linearRGB.rgb * vec3(12.92);

    return vec4(mix(higher, lower, cutoff), linearRGB.a);
}

vec4 toLinear(vec4 sRGB)
{
    bvec3 cutoff = lessThan(sRGB.rgb, vec3(0.04045));
    vec3 higher = pow((sRGB.rgb + vec3(0.055))/vec3(1.055), vec3(2.4));
    vec3 lower = sRGB.rgb/vec3(12.92);

    return vec4(mix(higher, lower, cutoff), sRGB.a);
}

void main() {
  const pbr_data_t pbr = primitives_pbr_data.entry[fs_in.primitive_id + fs_in.mtl_offset];

  vec3 base_color = pbr.base_color_factor.rgb * texture(g_s_colormaps[pbr.colormap_id], fs_in.uv).rgb;
  vec3 albedo     = pow(base_color, vec3(2.2));
  vec3 metallic_roughness = texture(g_s_metal_roughness_maps[pbr.metallic_roughness_id], fs_in.uv).rgb;
  float metallic  = pbr.metallic_factor * metallic_roughness.b;
  float roughness = pbr.roughness_factor * metallic_roughness.g;
  float ao        = 1.0;

  vec3 N = 
    // normalize(fs_in.normal);
  getNormalFromMap(texture(g_s_normal_maps[pbr.normal_id], fs_in.uv).rgb);

  vec3 V = normalize(lighting_data.eye_pos - fs_in.pos);
  vec3 R = reflect(-V, N);

  // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
  // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
  vec3 F0 = albedo;
  // vec3(0.04); 
  // F0 = mix(F0, albedo, metallic);

  // reflectance equation
  vec3 Lo = vec3(0.0);
  for(int i = 0; i < 1; ++i) 
    {
      // calculate per-light radiance
      const vec3 light_pos = vec3(0.0, 1000.0, 0.0);
      const vec3 kLightColor = vec3(1.0);
      // vec3(242.3/255.0, 224.7 / 255.0, 135.0 / 255.0);
      vec3 L = normalize(light_pos);
      // normalize(light_pos - fs_in.pos);
      vec3 H = normalize(V + L);
      // float distance = length(lightPositions[i] - WorldPos);
      // float attenuation = 1.0 / (distance * distance);
      vec3 radiance = kLightColor;
      // lightColors[i] * attenuation;

      // Cook-Torrance BRDF
      float NDF = DistributionGGX(N, H, roughness);   
      float G   = GeometrySmith(N, V, L, roughness);      
      vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
           
      vec3 numerator    = NDF * G * F; 
      float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
      vec3 specular = numerator / denominator;
        
      // kS is equal to Fresnel
      vec3 kS = F;
      // for energy conservation, the diffuse and specular light can't
      // be above 1.0 (unless the surface emits light); to preserve this
      // relationship the diffuse component (kD) should equal 1.0 - kS.
      vec3 kD = vec3(1.0) - kS;
      // multiply kD by the inverse metalness such that only non-metals 
      // have diffuse lighting, or a linear blend if partly metal (pure metals
      // have no diffuse light).
      kD *= 1.0 - metallic;	  

      // scale light by NdotL
      float NdotL = max(dot(N, L), 0.0);        

      // add to outgoing radiance Lo
      Lo += (kD * albedo / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    }   
    
  //
  // ambient lighting (we now use IBL as the ambient term)
  vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
  vec3 kS = F;
    
  vec3 kD = 1.0 - kS;
  kD *= 1.0 - metallic;	  

  vec3 irradiance = texture(s_irradiance[lighting_data.skybox], N).rgb;
  vec3 diffuse      = irradiance * albedo;

  // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
  const float MAX_REFLECTION_LOD = 8.0;
  vec3 prefilteredColor = textureLod(s_prefiltered[lighting_data.skybox], R,  roughness * MAX_REFLECTION_LOD).rgb;    
  vec2 brdf  = texture(s_brdf_lut[lighting_data.skybox], vec2(max(dot(N, V), 0.0), roughness)).rg;
  vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

  vec3 ambient = (kD * diffuse + specular) * ao;
    
  vec3 color = ambient + Lo;

  // HDR tonemapping
  color = color / (color + vec3(1.0));
  // gamma correct
  color = pow(color, vec3(1.0/2.2)); 

  FinalFragColor = vec4(color, 1.0);
}
