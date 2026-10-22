#version 450

layout(location = 0) in vec3 in_position;
layout(location = 2) in vec4 in_color;

layout(location = 0) flat out vec4 out_color;

// Screen-space overlay: positions are already in clip space with w = 1, as in most HUDs.
void main() {
    out_color = in_color;
    gl_Position = vec4(in_position.xy, 0.0, 1.0);
}
