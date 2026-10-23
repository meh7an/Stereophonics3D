#version 450

// Eight bytes per glyph, one per row (top first, bit 0 leftmost), four bytes per word.
layout(set = 0, binding = 0) uniform Font {
    uvec4 words[48];
} font;

layout(location = 0) flat in uint glyph;
layout(location = 1) flat in vec4 color;
layout(location = 2) in vec2 texel;

layout(location = 0) out vec4 out_color;

// Unlit pixels form the panel behind the text.
const vec4 BACKGROUND = vec4(0.0, 0.0, 0.0, 0.65);

void main() {
    uvec2 pixel = min(uvec2(texel), uvec2(7u));
    uint byte_index = glyph * 8u + pixel.y;
    uint word = font.words[byte_index / 16u][(byte_index / 4u) % 4u];
    uint row = (word >> (8u * (byte_index % 4u))) & 0xFFu;
    out_color = ((row >> pixel.x) & 1u) != 0u ? color : BACKGROUND;
}
