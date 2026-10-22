// Procedural environment shared by the skybox and the reflector: a vertical gradient with
// narrow bright bands (7.5 degrees of azimuth each) down to just below the horizon, so a
// reflection always shows band edges whose position is easy to measure.
vec3 environment(vec3 direction) {
    vec3 d = normalize(direction);
    vec3 sky = mix(vec3(0.25, 0.2, 0.15), vec3(0.4, 0.6, 0.95), d.y * 0.5 + 0.5);
    float band = step(0.5, fract(atan(d.z, d.x) * 48.0 / 6.2831853));
    return mix(sky, vec3(0.95), band * step(-0.3, d.y) * 0.6);
}
