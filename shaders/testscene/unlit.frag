#version 450
#extension GL_GOOGLE_include_directive : require

#include "mesh_in.glsl"

layout(location = 0) out vec4 out_color;

void main() {
    out_color = in_color;
}
