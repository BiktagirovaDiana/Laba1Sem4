//
//  Phong.metal
//  KG_1.1
//
//  Created by Macbook on 21.02.2026.
//

#include <metal_stdlib>
using namespace metal;

struct VertexIn
{
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

struct CameraCB
{
    float4x4 world;
    float4x4 view;
    float4x4 proj;
    float3   lightDir;
    float    lightIntensity;
    float3   lightColor;
    float    pad0;
    float3   cameraPos;
    float    timeSeconds;
};

struct MaterialCB
{
    float4 kd_ns;
    float4 ks_alpha;
    float2 uvScale;
    float2 uvSpeed;
    uint4 textureFlags;
    float4 detailParams;
};

struct VSOut
{
    float4 position [[position]];
    float3 worldPos;
    float3 worldN;
    float2 uv;
};

struct GBufferOut
{
    float4 albedo [[color(0)]];
    float4 normal [[color(1)]];
    float4 position [[color(2)]];
    float4 material [[color(3)]];
};

static float3x3 CotangentFrame(float3 N, float3 worldPos, float2 uv)
{
    const float3 dp1 = dfdx(worldPos);
    const float3 dp2 = dfdy(worldPos);
    const float2 duv1 = dfdx(uv);
    const float2 duv2 = dfdy(uv);

    const float3 dp2perp = cross(dp2, N);
    const float3 dp1perp = cross(N, dp1);
    float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    const float invMax = rsqrt(max(dot(T, T), dot(B, B)));
    return float3x3(T * invMax, B * invMax, N);
}

static float3 ApplyNormalMap(float3 baseNormal,
                             float3 worldPos,
                             float2 uv,
                             float3 normalSample,
                             float normalStrength)
{
    const float3 tangentNormal = normalize(float3(normalSample.xy * 2.0 - 1.0,
                                                  max(normalSample.z * 2.0 - 1.0, 0.0)));
    const float3 blendedTangentNormal =
        normalize(float3(tangentNormal.xy * normalStrength, tangentNormal.z));
    const float3x3 tbn = CotangentFrame(baseNormal, worldPos, uv);
    return normalize(tbn * blendedTangentNormal);
}

vertex VSOut vs_gbuffer(VertexIn vin [[stage_in]],
                        constant CameraCB& cb [[buffer(1)]],
                        constant MaterialCB& mat [[buffer(2)]],
                        texture2d<float> heightTex [[texture(0)]],
                        sampler linearSampler [[sampler(0)]])
{
    VSOut o;
    const float2 uv = vin.uv * mat.uvScale + mat.uvSpeed * cb.timeSeconds;
    float3 displacedPosition = vin.position;
    if (mat.textureFlags.z != 0u)
    {
        const float height = heightTex.sample(linearSampler, uv, level(0.0)).r;
        displacedPosition.y += height * mat.detailParams.x;
    }

    float4 wp = cb.world * float4(displacedPosition, 1.0);
    float4 vp = cb.view  * wp;
    o.position = cb.proj * vp;

    o.worldPos = wp.xyz;
    o.worldN = normalize((cb.world * float4(vin.normal, 0.0)).xyz);
    o.uv = uv;
    return o;
}

fragment GBufferOut ps_gbuffer(VSOut in [[stage_in]],
                               constant CameraCB& cb [[buffer(0)]],
                               constant MaterialCB& mat [[buffer(1)]],
                               texture2d<float> diffuseTex [[texture(0)]],
                               texture2d<float> normalTex [[texture(1)]],
                               sampler linearSampler [[sampler(0)]])
{
    float3 baseColor = mat.kd_ns.rgb;
    if (mat.textureFlags.x != 0u)
    {
        baseColor *= diffuseTex.sample(linearSampler, in.uv).rgb;
    }

    float3 worldNormal = normalize(in.worldN);
    if (mat.textureFlags.y != 0u)
    {
        const float3 normalSample = normalTex.sample(linearSampler, in.uv).rgb;
        worldNormal = ApplyNormalMap(worldNormal, in.worldPos, in.uv, normalSample, mat.detailParams.y);
    }

    GBufferOut outData;
    outData.albedo = float4(baseColor, 1.0);
    outData.normal = float4(worldNormal, 1.0);
    outData.position = float4(in.worldPos, 1.0);
    outData.material = float4(mat.ks_alpha.rgb, max(mat.kd_ns.w, 1.0));
    return outData;
}

struct FullscreenOut
{
    float4 position [[position]];
    float2 uv;
};

vertex FullscreenOut vs_fullscreen(uint vid [[vertex_id]])
{
    FullscreenOut o;

    float2 pos[3] =
    {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };

    o.position = float4(pos[vid], 0.0, 1.0);
    o.uv = pos[vid] * 0.5 + 0.5;
    o.uv.y = 1.0 - o.uv.y;
    return o;
}

fragment float4 ps_lighting(FullscreenOut in [[stage_in]],
                            constant CameraCB& cb [[buffer(0)]],
                            texture2d<float> gbufferAlbedo [[texture(0)]],
                            texture2d<float> gbufferNormal [[texture(1)]],
                            texture2d<float> gbufferPosition [[texture(2)]],
                            texture2d<float> gbufferMaterial [[texture(3)]],
                            sampler linearSampler [[sampler(0)]])
{
    const float2 uv = clamp(in.uv, float2(0.0), float2(1.0));
    const float4 albedoSample = gbufferAlbedo.sample(linearSampler, uv);
    if (albedoSample.a < 0.5)
    {
        float3 topColor = float3(0.5, 0.7, 1.0);
        float3 bottomColor = float3(0.7, 0.85, 1.0);
        float t = saturate(1.0 - uv.y);
        return float4(mix(bottomColor, topColor, t), 1.0);
    }

    const float3 N = normalize(gbufferNormal.sample(linearSampler, uv).xyz);
    const float3 worldPos = gbufferPosition.sample(linearSampler, uv).xyz;
    const float4 materialSample = gbufferMaterial.sample(linearSampler, uv);

    float3 L = normalize(-cb.lightDir);
    float3 V = normalize(cb.cameraPos - worldPos);
    float3 R = reflect(-L, N);

    const float ambient = 0.10;
    const float diff = max(dot(N, L), 0.0);
    const float spec = pow(max(dot(R, V), 0.0), materialSample.a);
    const float3 directionalRadiance = cb.lightColor * cb.lightIntensity;

    float3 color =
        albedoSample.rgb * ambient +
        albedoSample.rgb * diff * directionalRadiance +
        materialSample.rgb * spec * directionalRadiance;

    return float4(color, 1.0);
}
