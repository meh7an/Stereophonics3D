#version 450
#extension GL_GOOGLE_include_directive : require

#include "frame.glsl"

layout(location = 0) in vec3 in_position;

layout(location = 0) out vec3 out_direction;

// The common engine trick: a unit cube around the camera with z = w, so it lands on the far
// plane. Its w stays near 1, which is why a stereo shear needs the `sky` override.
void main() {
    out_direction = in_position;
    gl_Position = (frame.sky_view_proj * vec4(in_position, 1.0)).xyww;
}
