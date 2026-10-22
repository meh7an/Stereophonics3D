// Position written one component at a time, unoptimised, SPIR-V 1.6.
#version 450

layout(location = 0) in vec3 in_position;

void main() {
    gl_Position.x = in_position.x;
    gl_Position.y = in_position.y;
    gl_Position.zw = vec2(in_position.z, 1.0);
}
