#version 450
#extension GL_GOOGLE_include_directive : require

#include "env.glsl"
#include "frame.glsl"
#include "mesh_in.glsl"

layout(location = 0) out vec4 out_color;

// Mirror-like reflector: the reflected direction depends on the camera position constant,
// so each eye sees the environment shifted by its own offset (G5).
void main() {
    vec3 n = normalize(in_normal);
    vec3 v = normalize(frame.camera_position.xyz - in_world);
    vec3 reflected = environment(reflect(-v, n));
    out_color = vec4(mix(in_color.rgb, reflected, 0.85), 1.0);
}
