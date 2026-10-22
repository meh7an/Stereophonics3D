// Fragment shader: no Position output at all.
#version 450

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(1.0, 0.0, 1.0, 1.0);
}
