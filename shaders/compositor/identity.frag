#version 450

layout(set = 0, binding = 0) uniform sampler2D source;

layout(location = 0) out vec4 out_color;

// texelFetch, not texture: one source texel per output pixel with no filtering, so UNORM
// images pass through bit-exact.
void main() {
    out_color = texelFetch(source, ivec2(gl_FragCoord.xy), 0);
}
