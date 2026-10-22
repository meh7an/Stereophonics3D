#version 450
#extension GL_GOOGLE_include_directive : require

#include "frame.glsl"

layout(push_constant) uniform Draw {
    mat4 model;
} draw;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec2 in_uv;

layout(location = 0) out vec3 out_world;
layout(location = 1) out vec3 out_normal;
layout(location = 2) flat out vec4 out_color;
layout(location = 3) out vec2 out_uv;

void main() {
    vec4 world = draw.model * vec4(in_position, 1.0);
    out_world = world.xyz;
    out_normal = mat3(draw.model) * in_normal;
    out_color = in_color;
    out_uv = in_uv;
    gl_Position = frame.view_proj * world;
}
