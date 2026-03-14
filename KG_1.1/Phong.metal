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
    float3   pointLightPos;
    float    pointLightRange;
    float3   pointLightColor;
    float    pointLightIntensity;
    float3   spotLightPos;
    float    spotLightRange;
    float3   spotLightDir;
    float    spotLightConeAngle;
    float3   spotLightColor;
    float    spotLightIntensity;
};

struct MaterialCB
{
    float4 kd_ns;
    float4 ks_alpha;
    float2 uvScale;
    float2 uvSpeed;
    uint useTexture;
    float3 pad;
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

vertex VSOut vs_gbuffer(VertexIn vin [[stage_in]],
                        constant CameraCB& cb [[buffer(1)]])
{
    VSOut o;
    float4 wp = cb.world * float4(vin.position, 1.0);
    float4 vp = cb.view  * wp;
    o.position = cb.proj * vp;

    o.worldPos = wp.xyz;
    o.worldN = normalize((cb.world * float4(vin.normal, 0.0)).xyz);
    o.uv = vin.uv;
    return o;
}

fragment GBufferOut ps_gbuffer(VSOut in [[stage_in]],
                               constant CameraCB& cb [[buffer(0)]],
                               constant MaterialCB& mat [[buffer(1)]],
                               texture2d<float> diffuseTex [[texture(0)]],
                               sampler linearSampler [[sampler(0)]])
{
    float2 uv = in.uv * mat.uvScale + mat.uvSpeed * cb.timeSeconds;

    float3 baseColor = mat.kd_ns.rgb;
    if (mat.useTexture != 0)
    {
        baseColor *= diffuseTex.sample(linearSampler, uv).rgb;
    }

    GBufferOut outData;
    outData.albedo = float4(baseColor, 1.0);
    outData.normal = float4(normalize(in.worldN), 1.0);
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

static float LightMarkerMask(float3 worldPosition,
                             constant CameraCB& cb,
                             float2 uv,
                             float radius)
{
    const float4 clipPos = cb.proj * (cb.view * float4(worldPosition, 1.0));
    if (clipPos.w <= 0.0001)
    {
        return 0.0;
    }

    const float2 ndc = clipPos.xy / clipPos.w;
    const float2 markerUv = float2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    const float dist = distance(uv, markerUv);
    return 1.0 - smoothstep(radius * 0.65, radius, dist);
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

    const float3 toPointLight = cb.pointLightPos - worldPos;
    const float pointDistance = length(toPointLight);
    if (pointDistance < cb.pointLightRange)
    {
        const float3 pointL = toPointLight / max(pointDistance, 0.0001);
        const float pointAttenuation = pow(saturate(1.0 - pointDistance / cb.pointLightRange), 2.0);
        const float pointDiffuse = max(dot(N, pointL), 0.0);
        const float3 pointR = reflect(-pointL, N);
        const float pointSpecular = pow(max(dot(pointR, V), 0.0), materialSample.a);
        const float3 pointRadiance = cb.pointLightColor * cb.pointLightIntensity * pointAttenuation;

        color += albedoSample.rgb * pointDiffuse * pointRadiance;
        color += materialSample.rgb * pointSpecular * pointRadiance;
    }

    const float3 toSpotLight = cb.spotLightPos - worldPos;
    const float spotDistance = length(toSpotLight);
    if (spotDistance < cb.spotLightRange)
    {
        const float3 spotL = toSpotLight / max(spotDistance, 0.0001);
        const float coneLimit = cos(cb.spotLightConeAngle);
        const float coneDot = dot(normalize(-cb.spotLightDir), spotL);
        if (coneDot > coneLimit)
        {
            const float coneAttenuation = saturate((coneDot - coneLimit) / max(1.0 - coneLimit, 0.0001));
            const float distanceAttenuation = pow(saturate(1.0 - spotDistance / cb.spotLightRange), 2.0);
            const float spotAttenuation = coneAttenuation * distanceAttenuation;
            const float spotDiffuse = max(dot(N, spotL), 0.0);
            const float3 spotR = reflect(-spotL, N);
            const float spotSpecular = pow(max(dot(spotR, V), 0.0), materialSample.a);
            const float3 spotRadiance = cb.spotLightColor * cb.spotLightIntensity * spotAttenuation;

            color += albedoSample.rgb * spotDiffuse * spotRadiance;
            color += materialSample.rgb * spotSpecular * spotRadiance;
        }
    }

    const float pointMarker = LightMarkerMask(cb.pointLightPos, cb, uv, 0.022);
    const float spotMarker = LightMarkerMask(cb.spotLightPos, cb, uv, 0.024);
    color = mix(color, float3(0.2, 1.0, 0.2), pointMarker);
    color = mix(color, float3(1.0, 0.1, 0.1), spotMarker);

    return float4(color, 1.0);
}
