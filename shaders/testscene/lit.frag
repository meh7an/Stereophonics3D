#version 450
#extension GL_GOOGLE_include_directive : require

#include "frame.glsl"
#include "mesh_in.glsl"

layout(set = 0, binding = 1) uniform sampler2DShadow shadow_map;

layout(location = 0) out vec4 out_color;

float shadow_factor() {
    if (frame.light_direction.w < 0.5) {
        return 1.0;
    }
    vec4 light_clip = frame.light_view_proj * vec4(in_world, 1.0);
    vec3 light_ndc = light_clip.xyz / light_clip.w;
    return texture(shadow_map, vec3(light_ndc.xy * 0.5 + 0.5, light_ndc.z));
}

// Blinn-Phong with the specular term computed from the camera position constant, so the
// highlight moves with the eye (G5).
void main() {
    vec3 n = normalize(in_normal);
    vec3 l = normalize(frame.light_direction.xyz);
    vec3 v = normalize(frame.camera_position.xyz - in_world);
    vec3 h = normalize(l + v);
    float n_dot_l = max(dot(n, l), 0.0);
    float shadow = shadow_factor();
    float specular = pow(max(dot(n, h), 0.0), 64.0) * step(0.0, dot(n, l)) * shadow;
    vec3 light = frame.light_color.rgb;
    vec3 rgb = in_color.rgb * (frame.light_color.a + n_dot_l * shadow * light);
    out_color = vec4(rgb + specular * light, 1.0);
}
