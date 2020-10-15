
#include <metal_stdlib>
using namespace metal;

#include "teapot.h"

struct VertexUniforms {
    float4x4 modelViewMatrix;
    float4x4 projectionMatrix;
};

struct quadTessellationFactorsHalf {
    half edgeTessellationFactor[4];
    half insideTessellationFactor[2];
};

kernel void compute_tess_factors(device quadTessellationFactorsHalf *factors [[buffer(0)]],
                                 uint patchIndex [[thread_position_in_grid]])
{
    device quadTessellationFactorsHalf *patchFactors = factors + patchIndex;
    patchFactors->edgeTessellationFactor[0] = 8.0h;
    patchFactors->edgeTessellationFactor[1] = 8.0h;
    patchFactors->edgeTessellationFactor[2] = 8.0h;
    patchFactors->edgeTessellationFactor[3] = 8.0h;
    patchFactors->insideTessellationFactor[0] = 8.0h;
    patchFactors->insideTessellationFactor[1] = 8.0h;
}

kernel void compute_control_point_position(device float3 *positions [[buffer(0)]],
                                           uint index [[thread_position_in_grid]])
{
    positions[index] = float3(teapotData[index * 3 + 0],
                              teapotData[index * 3 + 1],
                              teapotData[index * 3 + 2]);
}

struct ControlPoint {
    float3 position [[attribute(0)]];
};

struct PatchIn {
    patch_control_point<ControlPoint> controlPoints;
};

struct VertexOut {
    float4 position [[position]];
    float3 eyePosition;
};

[[patch(quad, 16)]]
vertex VertexOut cub_bez_vertex(PatchIn patch                     [[stage_in]],
                                constant VertexUniforms &uniforms [[buffer(1)]],
                                float2 positionInPatch            [[position_in_patch]])
{
    float3 c00 = patch.controlPoints[ 0].position;
    float3 c01 = patch.controlPoints[ 1].position;
    float3 c02 = patch.controlPoints[ 2].position;
    float3 c03 = patch.controlPoints[ 3].position;
    float3 c10 = patch.controlPoints[ 4].position;
    float3 c11 = patch.controlPoints[ 5].position;
    float3 c12 = patch.controlPoints[ 6].position;
    float3 c13 = patch.controlPoints[ 7].position;
    float3 c20 = patch.controlPoints[ 8].position;
    float3 c21 = patch.controlPoints[ 9].position;
    float3 c22 = patch.controlPoints[10].position;
    float3 c23 = patch.controlPoints[11].position;
    float3 c30 = patch.controlPoints[12].position;
    float3 c31 = patch.controlPoints[13].position;
    float3 c32 = patch.controlPoints[14].position;
    float3 c33 = patch.controlPoints[15].position;
    
    float2 uv = positionInPatch;
    
    float4x4 M(
        1.0f, -3.0f,  3.0f, -1.0f,
        0.0f,  3.0f, -6.0f,  3.0f,
        0.0f,  0.0f,  3.0f, -3.0f,
        0.0f,  0.0f,  0.0f,  1.0f
    );
    
    float4x4 Mt = transpose(M);

    float4x4 Gx(
        c00.x, c10.x, c20.x, c30.x,
        c01.x, c11.x, c21.x, c31.x,
        c02.x, c12.x, c22.x, c32.x,
        c03.x, c13.x, c23.x, c33.x
    );

    float4x4 Gy(
        c00.y, c10.y, c20.y, c30.y,
        c01.y, c11.y, c21.y, c31.y,
        c02.y, c12.y, c22.y, c32.y,
        c03.y, c13.y, c23.y, c33.y
    );

    float4x4 Gz(
        c00.z, c10.z, c20.z, c30.z,
        c01.z, c11.z, c21.z, c31.z,
        c02.z, c12.z, c22.z, c32.z,
        c03.z, c13.z, c23.z, c33.z
    );
    
    float4 U { 1.0f, uv.x, uv.x * uv.x, uv.x * uv.x * uv.x };
    float4 V { 1.0f, uv.y, uv.y * uv.y, uv.y * uv.y * uv.y };
    
    float x = dot(U, M * Gx * Mt * V);
    float y = dot(U, M * Gy * Mt * V);
    float z = dot(U, M * Gz * Mt * V);
    
    float4 modelPosition(x, y, z, 1.0f);
    
    float4 eyePosition = uniforms.modelViewMatrix * modelPosition;

    VertexOut out;
    out.position = uniforms.projectionMatrix * eyePosition;
    out.eyePosition = eyePosition.xyz;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]])
{
    float3 dfdxPos = dfdx(in.eyePosition);
    float3 dfdyPos = dfdy(in.eyePosition);
    float3 N = normalize(cross(dfdxPos, dfdyPos));
    float4 color = float4(N * 0.5f + 0.5f, 1.0f);
    return color;
}
