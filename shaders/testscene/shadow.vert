#version 450
#extension GL_GOOGLE_include_directive : require

#include "frame.glsl"

layout(push_constant) uniform Draw {
    mat4 model;
} draw;

layout(location = 0) in vec3 in_position;

// Depth-only: no fragment shader, rendered from the light into a square target.
void main() {
    gl_Position = frame.light_view_proj * draw.model * vec4(in_position, 1.0);
}
