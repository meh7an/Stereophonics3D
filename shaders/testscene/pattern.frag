#version 450
#extension GL_GOOGLE_include_directive : require

#include "mesh_in.glsl"

layout(location = 0) out vec4 out_color;

// Fine magenta/green checker: high-frequency detail that cannot be reconstructed from
// neighbouring pixels, which is what G1 looks for in the revealed strip.
void main() {
    uvec2 cell = uvec2(floor(in_uv));
    bool odd = ((cell.x + cell.y) & 1u) == 1u;
    out_color = odd ? vec4(0.0, 1.0, 0.0, 1.0) : in_color;
}
