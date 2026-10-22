// Implicit gl_PerVertex, one whole-vector store, optimised, SPIR-V 1.0.
#version 450

layout(set = 0, binding = 0) uniform Camera {
    mat4 view_proj;
} camera;

layout(location = 0) in vec3 in_position;

void main() {
    gl_Position = camera.view_proj * vec4(in_position, 1.0);
}
