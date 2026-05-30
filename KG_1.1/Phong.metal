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
    float4   postProcessParams;
};

struct ShadowCB
{
    float4x4 lightViewProj[4];
    float4 cascadeSplits;
    float4 texelSizes;
    float4 params;
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

struct StructuredBufferElement
{
    uint value;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct ParticleAnimationCB
{
    float4 centerAndFactor;
    uint instanceCount;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct DustAnimationCB
{
    float4 centerAndTime;
    float4 motionParams;
    uint instanceCount;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct RainAnimationCB
{
    float4 centerAndTime;
    float4 volumeAndSpeed;
    float4 bounceParams;
    uint instanceCount;
    uint collisionPlaneCount;
    uint pad0;
    uint pad1;
};

struct RainCollisionPlane
{
    float4 minXZMaxXZ;
    float4 yAndPadding;
};

struct ParticleInstance
{
    float4 baseCenterAndSize;
    float4 animatedCenterAndSeed;
};

static float Hash01(float value)
{
    return fract(sin(value * 12.9898) * 43758.5453);
}

static float3 DustDirection(uint planeIndex)
{
    const float fi = float(planeIndex) + 1.0;
    const float z = Hash01(fi * 3.17) * 2.0 - 1.0;
    const float angle = Hash01(fi * 7.91) * 6.28318530718;
    const float xyRadius = sqrt(max(0.0, 1.0 - z * z));
    return normalize(float3(cos(angle) * xyRadius, sin(angle) * xyRadius, z));
}

kernel void cs_update_particles(const device ParticleInstance* baseInstances [[buffer(0)]],
                                device ParticleInstance* animatedInstances [[buffer(1)]],
                                constant ParticleAnimationCB& cb [[buffer(2)]],
                                uint instanceId [[thread_position_in_grid]])
{
    if (instanceId >= cb.instanceCount)
    {
        return;
    }

    const float3 center = cb.centerAndFactor.xyz;
    const float sphereFactor = cb.centerAndFactor.w;
    const ParticleInstance baseInstance = baseInstances[instanceId];
    const float3 baseCenter = baseInstance.baseCenterAndSize.xyz;
    const float3 animatedCenter = center + (baseCenter - center) * sphereFactor;

    ParticleInstance animatedInstance = baseInstance;
    animatedInstance.animatedCenterAndSeed.xyz = animatedCenter;
    animatedInstances[instanceId] = animatedInstance;
}

kernel void cs_update_dust_particles(const device ParticleInstance* baseInstances [[buffer(0)]],
                                     device ParticleInstance* animatedInstances [[buffer(1)]],
                                     constant DustAnimationCB& cb [[buffer(2)]],
                                     uint instanceId [[thread_position_in_grid]])
{
    if (instanceId >= cb.instanceCount)
    {
        return;
    }

    const float fi = float(instanceId) + 1.0;
    const float3 driftDir = DustDirection(instanceId);
    const float3 center = cb.centerAndTime.xyz;
    const float time = cb.centerAndTime.w;
    const float driftAmplitude = cb.motionParams.x;
    const float driftSpeed = cb.motionParams.y;
    const float swirlAmplitude = cb.motionParams.z;

    const float phaseA = Hash01(fi * 11.3) * 6.28318530718;
    const float phaseB = Hash01(fi * 19.7) * 6.28318530718;

    const ParticleInstance baseInstance = baseInstances[instanceId];
    const float3 baseCenter = baseInstance.baseCenterAndSize.xyz;
    const float3 fromCenter =
        normalize(baseCenter - center + driftDir * 0.001);
    const float3 swirlDir = normalize(cross(driftDir, fromCenter) + DustDirection(instanceId + 137u) * 0.15);

    const float drift = sin(time * driftSpeed + phaseA) * driftAmplitude;
    const float hover = cos(time * driftSpeed * 1.7 + phaseB) * swirlAmplitude;
    const float3 offset = driftDir * drift + swirlDir * hover;

    ParticleInstance animatedInstance = baseInstance;
    animatedInstance.animatedCenterAndSeed.xyz = baseCenter + offset;
    animatedInstances[instanceId] = animatedInstance;
}

kernel void cs_update_rain_particles(const device ParticleInstance* baseInstances [[buffer(0)]],
                                     device ParticleInstance* animatedInstances [[buffer(1)]],
                                     constant RainAnimationCB& cb [[buffer(2)]],
                                     const device RainCollisionPlane* collisionPlanes [[buffer(3)]],
                                     uint instanceId [[thread_position_in_grid]])
{
    if (instanceId >= cb.instanceCount)
    {
        return;
    }

    const ParticleInstance baseInstance = baseInstances[instanceId];
    const float3 center = cb.centerAndTime.xyz;
    const float time = cb.centerAndTime.w;
    const float radius = cb.volumeAndSpeed.x;
    const float fallHeight = max(cb.volumeAndSpeed.y, 0.001);
    const float fallSpeed = cb.volumeAndSpeed.z;
    const float bounceHeight = max(cb.bounceParams.x, 0.0);
    const float respawnDistance = max(cb.bounceParams.y, 0.0);
    const float collisionBias = max(cb.bounceParams.z, 0.0);
    const float topY = center.y + radius;
    const float2 xz = baseInstance.baseCenterAndSize.xz;
    const float particleVisualLift = max(baseInstance.baseCenterAndSize.w, 0.001);
    const float startOffset = clamp(topY - baseInstance.baseCenterAndSize.y, 0.0, fallHeight);

    float collisionY = topY - fallHeight;
    for (uint i = 0; i < cb.collisionPlaneCount; ++i)
    {
        const RainCollisionPlane plane = collisionPlanes[i];
        const float4 bounds = plane.minXZMaxXZ;
        if (xz.x < bounds.x || xz.x > bounds.z || xz.y < bounds.y || xz.y > bounds.w)
        {
            continue;
        }

        const float candidateY = plane.yAndPadding.x;
        if (candidateY < topY && candidateY > collisionY)
        {
            collisionY = candidateY;
        }
    }

    collisionY += particleVisualLift + collisionBias;
    collisionY = min(collisionY, topY);

    const float impactDistance = clamp(topY - collisionY, 0.0, fallHeight);
    const float safeFallSpeed = max(fallSpeed, 0.001);
    const float fallDuration = impactDistance / safeFallSpeed;
    const float bounceDuration = (bounceHeight > 0.0)
        ? max(respawnDistance / safeFallSpeed, 0.06)
        : 0.0;
    const float respawnDelay = max(respawnDistance / safeFallSpeed, 0.0);
    const float cycleDuration = max(fallDuration + bounceDuration + respawnDelay, 0.001);
    const float cycleTime = fmod(startOffset / safeFallSpeed + time, cycleDuration);

    float3 animatedCenter = baseInstance.baseCenterAndSize.xyz;
    if (cycleTime < fallDuration)
    {
        animatedCenter.y = topY - cycleTime * safeFallSpeed;
    }
    else if (cycleTime < fallDuration + bounceDuration)
    {
        const float bounceTime = (cycleTime - fallDuration) / max(bounceDuration, 0.001);
        const float bounceArc = 1.0 - pow(2.0 * bounceTime - 1.0, 2.0);
        animatedCenter.y = collisionY + bounceHeight * max(bounceArc, 0.0);
    }
    else
    {
        animatedCenter = float3(xz.x, collisionY - fallHeight - bounceHeight - 10.0, xz.y);
    }

    ParticleInstance animatedInstance = baseInstance;
    animatedInstance.animatedCenterAndSeed.xyz = animatedCenter;
    animatedInstances[instanceId] = animatedInstance;
}

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
        // Treat 0.5 as the neutral height and displace along the surface normal.
        const float height = heightTex.sample(linearSampler, uv, level(0.0)).r - 0.5;
        const float3 objectNormal = normalize(vin.normal);
        displacedPosition += objectNormal * (height * mat.detailParams.x);
    }

    float4 wp = cb.world * float4(displacedPosition, 1.0);
    float4 vp = cb.view  * wp;
    o.position = cb.proj * vp;

    o.worldPos = wp.xyz;
    o.worldN = normalize((cb.world * float4(vin.normal, 0.0)).xyz);
    o.uv = uv;
    return o;
}

vertex float4 vs_shadow(VertexIn vin [[stage_in]],
                        constant CameraCB& cb [[buffer(1)]],
                        constant MaterialCB& mat [[buffer(2)]],
                        texture2d<float> heightTex [[texture(0)]],
                        sampler linearSampler [[sampler(0)]])
{
    const float2 uv = vin.uv * mat.uvScale + mat.uvSpeed * cb.timeSeconds;
    float3 displacedPosition = vin.position;
    if (mat.textureFlags.z != 0u)
    {
        const float height = heightTex.sample(linearSampler, uv, level(0.0)).r - 0.5;
        displacedPosition += normalize(vin.normal) * (height * mat.detailParams.x);
    }

    const float4 wp = cb.world * float4(displacedPosition, 1.0);
    return cb.proj * cb.view * wp;
}

// ── Fence plane: shadow pass with alpha-test ──────────────────────────────────
// VSOut is reused — we need uv in the fragment stage.
struct FenceShadowVSOut
{
    float4 position [[position]];
    float2 uv;
};

vertex FenceShadowVSOut vs_fence_shadow(VertexIn vin [[stage_in]],
                                        constant CameraCB& cb [[buffer(1)]],
                                        constant MaterialCB& mat [[buffer(2)]])
{
    const float4 wp = cb.world * float4(vin.position, 1.0);
    FenceShadowVSOut o;
    o.position = cb.proj * cb.view * wp;
    o.uv = vin.uv * mat.uvScale;
    return o;
}

fragment void ps_fence_shadow(FenceShadowVSOut in [[stage_in]],
                              texture2d<float> diffuseTex [[texture(0)]],
                              sampler linearSampler [[sampler(0)]])
{
    const float alpha = diffuseTex.sample(linearSampler, in.uv).a;
    if (alpha < 0.5)
    {
        discard_fragment();
    }
}

// ── Fence plane: gbuffer pass ─────────────────────────────────────────────────
vertex VSOut vs_fence_gbuffer(VertexIn vin [[stage_in]],
                              constant CameraCB& cb [[buffer(1)]],
                              constant MaterialCB& mat [[buffer(2)]])
{
    const float4 wp = cb.world * float4(vin.position, 1.0);
    VSOut o;
    o.position = cb.proj * cb.view * wp;
    o.worldPos = wp.xyz;
    o.worldN = normalize((cb.world * float4(vin.normal, 0.0)).xyz);
    o.uv = vin.uv * mat.uvScale;
    return o;
}

fragment GBufferOut ps_fence_gbuffer(VSOut in [[stage_in]],
                                     constant CameraCB& cb [[buffer(0)]],
                                     constant MaterialCB& mat [[buffer(1)]],
                                     texture2d<float> diffuseTex [[texture(0)]],
                                     sampler linearSampler [[sampler(0)]])
{
    const float4 diffuseSample = diffuseTex.sample(linearSampler, in.uv);
    if (diffuseSample.a < 0.5)
    {
        discard_fragment();
    }

    GBufferOut outData;
    outData.albedo   = float4(diffuseSample.rgb * mat.kd_ns.rgb, 1.0);
    outData.normal   = float4(normalize(in.worldN), 1.0);
    outData.position = float4(in.worldPos, 1.0);
    // Use standard Phong material channel (non-zero .a → full lighting)
    outData.material = float4(mat.ks_alpha.rgb, max(mat.kd_ns.w, 1.0));
    return outData;
}

vertex VSOut vs_particle_billboard(VertexIn vin [[stage_in]],
                                   constant CameraCB& cb [[buffer(1)]],
                                   constant MaterialCB& mat [[buffer(2)]],
                                   const device ParticleInstance* instances [[buffer(3)]],
                                   uint instanceId [[instance_id]])
{
    const ParticleInstance instance = instances[instanceId];
    const float3 center = instance.animatedCenterAndSeed.xyz;
    const float size = instance.baseCenterAndSize.w;

    float3 forward = normalize(cb.cameraPos - center);
    float3 upHint = float3(0.0, 1.0, 0.0);
    float3 right = cross(upHint, forward);
    if (dot(right, right) < 1e-5)
    {
        upHint = float3(1.0, 0.0, 0.0);
        right = cross(upHint, forward);
    }
    right = normalize(right);
    const float3 up = normalize(cross(forward, right));

    const float2 uv = vin.uv * mat.uvScale + mat.uvSpeed * cb.timeSeconds;
    const float3 worldPos = center + right * (vin.position.x * size) + up * (vin.position.y * size);
    const float4 wp = float4(worldPos, 1.0);
    const float4 vp = cb.view * wp;

    VSOut o;
    o.position = cb.proj * vp;
    o.worldPos = worldPos;
    o.worldN = forward;
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
    float alpha = 1.0;
    const bool isPlaneReplacement = (mat.textureFlags.w != 0u);
    if (mat.textureFlags.x != 0u)
    {
        const float4 diffuseSample = diffuseTex.sample(linearSampler, in.uv);
        baseColor *= diffuseSample.rgb;
        if (isPlaneReplacement)
        {
            alpha = mat.ks_alpha.a * diffuseSample.a;

            // Cut semi-transparent fringe pixels harder to avoid blue halos on the sprite edges.
            if (alpha < 0.85)
            {
                discard_fragment();
            }
        }
    }

    float3 worldNormal = normalize(in.worldN);
    if (mat.textureFlags.y != 0u)
    {
        const float3 normalSample = normalTex.sample(linearSampler, in.uv).rgb;
        worldNormal = ApplyNormalMap(worldNormal, in.worldPos, in.uv, normalSample, mat.detailParams.y);
    }

    GBufferOut outData;
    outData.albedo = float4(baseColor, alpha);
    outData.normal = float4(worldNormal, 1.0);
    outData.position = float4(in.worldPos, 1.0);
    outData.material = isPlaneReplacement
        ? float4(0.0, 0.0, 0.0, 0.0)
        : float4(mat.ks_alpha.rgb, max(mat.kd_ns.w, 1.0));
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

    constexpr float2 pos[4] =
    {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };

    o.position = float4(pos[vid], 0.0, 1.0);
    o.uv = pos[vid] * 0.5 + 0.5;
    o.uv.y = 1.0 - o.uv.y;
    return o;
}

fragment float4 ps_gbuffer_stub(FullscreenOut in [[stage_in]],
                                texture2d<float> gbufferAlbedo [[texture(0)]],
                                texture2d<float> gbufferNormal [[texture(1)]],
                                texture2d<float> gbufferPosition [[texture(2)]],
                                texture2d<float> gbufferMaterial [[texture(3)]],
                                sampler linearSampler [[sampler(0)]])
{
    const float2 uv = clamp(in.uv, float2(0.0), float2(1.0));

    const float4 albedo = gbufferAlbedo.sample(linearSampler, uv);
    const float4 packedNormal = gbufferNormal.sample(linearSampler, uv);
    const float4 worldPosition = gbufferPosition.sample(linearSampler, uv);
    const float4 material = gbufferMaterial.sample(linearSampler, uv);

    const float3 normal = normalize(packedNormal.xyz * 2.0 - 1.0);
    const float specularPower = material.a;

    (void)worldPosition;
    (void)normal;
    (void)specularPower;

    return float4(albedo.rgb, 1.0);
}

static float SampleShadowMap(depth2d<float> shadowMap,
                             sampler shadowSampler,
                             float2 uv,
                             float compareDepth)
{
    return shadowMap.sample_compare(shadowSampler, uv, compareDepth);
}

static float SampleShadowPCF(depth2d<float> shadowMap,
                             sampler shadowSampler,
                             float2 uv,
                             float compareDepth,
                             float texelSize)
{
    float lit = 0.0;
    constexpr int radius = 1;
    constexpr float sampleCount = float((radius * 2 + 1) * (radius * 2 + 1));

    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            const float2 offset = float2(float(x), float(y)) * texelSize;
            lit += SampleShadowMap(shadowMap, shadowSampler, uv + offset, compareDepth);
        }
    }

    return lit / sampleCount;
}

static float3 ApplyVintagePostProcess(float3 color, float2 uv, float timeSeconds)
{
    const float luminance = dot(color, float3(0.299, 0.587, 0.114));
    const float3 sepia = float3(
        dot(color, float3(0.393, 0.769, 0.189)),
        dot(color, float3(0.349, 0.686, 0.168)),
        dot(color, float3(0.272, 0.534, 0.131)));

    color = mix(float3(luminance), color, 0.72);
    color = mix(color, sepia, 0.42);

    const float2 centeredUv = uv * 2.0 - 1.0;
    const float vignette = smoothstep(1.35, 0.25, dot(centeredUv, centeredUv));
    color *= mix(0.48, 1.08, vignette);

    const float scanline = sin((uv.y + timeSeconds * 0.03) * 900.0) * 0.018;
    const float grain = fract(sin(dot(uv + timeSeconds, float2(12.9898, 78.233))) * 43758.5453) - 0.5;
    color += scanline + grain * 0.055;

    return saturate(color);
}

fragment float4 ps_lighting(FullscreenOut in [[stage_in]],
                            constant CameraCB& cb [[buffer(0)]],
                            constant ShadowCB& shadowCb [[buffer(1)]],
                            texture2d<float> gbufferAlbedo [[texture(0)]],
                            texture2d<float> gbufferNormal [[texture(1)]],
                            texture2d<float> gbufferPosition [[texture(2)]],
                            texture2d<float> gbufferMaterial [[texture(3)]],
                            depth2d<float> shadowMap0 [[texture(4)]],
                            depth2d<float> shadowMap1 [[texture(5)]],
                            depth2d<float> shadowMap2 [[texture(6)]],
                            depth2d<float> shadowMap3 [[texture(7)]],
                            sampler linearSampler [[sampler(0)]],
                            sampler shadowSampler [[sampler(1)]])
{
    const float2 uv = clamp(in.uv, float2(0.0), float2(1.0));
    const float4 albedoSample = gbufferAlbedo.sample(linearSampler, uv);
    if (albedoSample.a < 0.5)
    {
        float3 topColor = float3(0.5, 0.7, 1.0);
        float3 bottomColor = float3(0.7, 0.85, 1.0);
        float t = saturate(1.0 - uv.y);
        float3 color = mix(bottomColor, topColor, t);
        if (cb.postProcessParams.x > 0.5)
        {
            color = ApplyVintagePostProcess(color, uv, cb.postProcessParams.y);
        }
        return float4(color, 1.0);
    }

    const float3 N = normalize(gbufferNormal.sample(linearSampler, uv).xyz);
    const float3 worldPos = gbufferPosition.sample(linearSampler, uv).xyz;
    const float4 materialSample = gbufferMaterial.sample(linearSampler, uv);
    if (materialSample.a < 0.5)
    {
        float3 color = albedoSample.rgb * 1.15;
        if (cb.postProcessParams.x > 0.5)
        {
            color = ApplyVintagePostProcess(color, uv, cb.postProcessParams.y);
        }
        return float4(color, 1.0);
    }

    float3 L = normalize(-cb.lightDir);
    float3 V = normalize(cb.cameraPos - worldPos);
    float3 R = reflect(-L, N);

    const float ambient = 0.10;
    const float diff = max(dot(N, L), 0.0);
    const float spec = pow(max(dot(R, V), 0.0), materialSample.a);
    const float3 directionalRadiance = cb.lightColor * cb.lightIntensity;
    const float viewDepth = -(cb.view * float4(worldPos, 1.0)).z;

    uint cascadeIndex = 0u;
    if (viewDepth > shadowCb.cascadeSplits.x) cascadeIndex = 1u;
    if (viewDepth > shadowCb.cascadeSplits.y) cascadeIndex = 2u;
    if (viewDepth > shadowCb.cascadeSplits.z) cascadeIndex = 3u;

    const float4 lightClip = shadowCb.lightViewProj[cascadeIndex] * float4(worldPos, 1.0);
    const float3 lightNdc = lightClip.xyz / lightClip.w;
    const float2 shadowUv = float2(lightNdc.x * 0.5 + 0.5, 1.0 - (lightNdc.y * 0.5 + 0.5));
    const float shadowDepth = lightNdc.z;
    float shadowVisibility = 1.0;

    if (shadowCb.params.w > 0.5 &&
        all(shadowUv >= float2(0.0)) &&
        all(shadowUv <= float2(1.0)) &&
        shadowDepth >= 0.0 &&
        shadowDepth <= 1.0)
    {
        const float normalBias = shadowCb.params.y * (1.0 - saturate(dot(N, L)));
        const float compareDepth = shadowDepth - shadowCb.params.x - normalBias;
        const float texel = shadowCb.texelSizes[cascadeIndex];

        float lit = 1.0;
        if (cascadeIndex == 0u)
        {
            lit = SampleShadowPCF(shadowMap0, shadowSampler, shadowUv, compareDepth, texel);
        }
        else if (cascadeIndex == 1u)
        {
            lit = SampleShadowPCF(shadowMap1, shadowSampler, shadowUv, compareDepth, texel);
        }
        else if (cascadeIndex == 2u)
        {
            lit = SampleShadowPCF(shadowMap2, shadowSampler, shadowUv, compareDepth, texel);
        }
        else
        {
            lit = SampleShadowPCF(shadowMap3, shadowSampler, shadowUv, compareDepth, texel);
        }

        shadowVisibility = mix(1.0 - shadowCb.params.z, 1.0, lit);
    }

    float3 color =
        albedoSample.rgb * ambient +
        (albedoSample.rgb * diff + materialSample.rgb * spec) * directionalRadiance * shadowVisibility;

    if (cb.postProcessParams.x > 0.5)
    {
        color = ApplyVintagePostProcess(color, uv, cb.postProcessParams.y);
    }

    return float4(color, 1.0);
}
