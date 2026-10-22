// DXC style: SV_Position becomes its own variable decorated BuiltIn Position.
cbuffer Camera : register(b0) {
    float4x4 view_proj;
};

float4 main(float3 position : POSITION) : SV_Position {
    return mul(view_proj, float4(position, 1.0));
}
