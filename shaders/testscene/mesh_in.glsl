// Fragment inputs matching every output of mesh.vert. Each mesh fragment shader declares
// all of them, used or not, so no vertex output is left unconsumed.
layout(location = 0) in vec3 in_world;
layout(location = 1) in vec3 in_normal;
layout(location = 2) flat in vec4 in_color;
layout(location = 3) in vec2 in_uv;
