#version 450

layout(set = 0, binding = 0) uniform sampler2D scene_color;

layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

// Set from `TONEMAP_EXPOSURE` in the Rust scene module, the single source of truth.
layout(constant_id = 0) const float EXPOSURE = 1.0;

// Reinhard on an exposed HDR target: a minimal post-process that samples the scene
// colour, which classification must keep per-eye.
void main() {
    vec3 hdr = texture(scene_color, in_uv).rgb * EXPOSURE;
    out_color = vec4(hdr / (1.0 + hdr), 1.0);
}
