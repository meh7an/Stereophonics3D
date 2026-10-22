// Per-frame constants, laid out like an engine's camera constant buffer so the stereo
// milestones can patch the camera position per eye (G5).
layout(set = 0, binding = 0) uniform Frame {
    mat4 view_proj;
    // Rotation-only view times projection, for the skybox.
    mat4 sky_view_proj;
    mat4 light_view_proj;
    vec4 camera_position;
    // xyz points toward the light; w is 1 when the shadow map holds this frame's shadows.
    vec4 light_direction;
    // rgb is the light colour; a is the ambient strength.
    vec4 light_color;
} frame;
