// DXC style with an output struct: members become separate interface variables.
struct VsOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

VsOut main(float3 position : POSITION, float2 uv : TEXCOORD0) {
    VsOut result;
    result.position = float4(position, 1.0);
    result.uv = uv;
    return result;
}
