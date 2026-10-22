// The application already owns a push block (bytes 0..64), as in the test scene; the
// stereo members must be appended to it rather than declared as a second block.
#version 450

layout(push_constant) uniform Draw {
    mat4 model;
} draw;

layout(location = 0) in vec3 in_position;

void main() {
    gl_Position = draw.model * vec4(in_position, 1.0);
}
