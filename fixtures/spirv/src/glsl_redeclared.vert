// gl_PerVertex redeclared with Position as its only member.
#version 450

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) in vec4 in_position;

void main() {
    gl_Position = in_position;
}
