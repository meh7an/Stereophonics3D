#version 450
#extension GL_GOOGLE_include_directive : require

#include "env.glsl"

layout(location = 0) in vec3 in_direction;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(environment(in_direction), 1.0);
}
