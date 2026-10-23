#version 450

// One instance per character cell, laid out by the layer's readout module.
layout(location = 0) in vec2 cell_origin;
layout(location = 1) in uint glyph;
layout(location = 2) in uint color;

layout(push_constant) uniform Push {
    vec2 extent;
    // Cell size in pixels: 8 font pixels times the readout scale.
    float cell;
} push;

layout(location = 0) flat out uint out_glyph;
layout(location = 1) flat out vec4 out_color;
// Position inside the glyph, 0 to 8 in font pixels.
layout(location = 2) out vec2 out_texel;

const vec2 CORNERS[6] = vec2[](
    vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
    vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0)
);

void main() {
    vec2 corner = CORNERS[gl_VertexIndex];
    vec2 pixel = cell_origin + corner * push.cell;
    gl_Position = vec4(pixel / push.extent * 2.0 - 1.0, 0.0, 1.0);
    out_glyph = glyph;
    out_color = unpackUnorm4x8(color);
    out_texel = corner * 8.0;
}
