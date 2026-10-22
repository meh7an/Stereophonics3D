#version 450

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
} pc;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) flat out vec3 out_color;

void main() {
    gl_Position = pc.view_proj * vec4(in_position, 1.0);
    out_color = in_color;
}
