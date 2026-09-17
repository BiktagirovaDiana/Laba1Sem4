#import "ShadowRenderer.hpp"
#import <Foundation/Foundation.h>

#include <cmath>

static simd::float4x4 LookAtRH(simd::float3 eye, simd::float3 at, simd::float3 up)
{
    simd::float3 z = simd::normalize(eye - at);
    simd::float3 x = simd::normalize(simd::cross(up, z));
    simd::float3 y = simd::cross(z, x);

    simd::float4x4 m;
    m.columns[0] = simd::float4{x.x, y.x, z.x, 0.0f};
    m.columns[1] = simd::float4{x.y, y.y, z.y, 0.0f};
    m.columns[2] = simd::float4{x.z, y.z, z.z, 0.0f};
    m.columns[3] = simd::float4{
        -simd::dot(x, eye),
        -simd::dot(y, eye),
        -simd::dot(z, eye),
         1.0f
    };
    return m;
}

static simd::float4x4 OrthoRH(float left, float right, float bottom, float top, float zn, float zf)
{
    simd::float4x4 m = {};
    m.columns[0] = simd::float4{2.0f / (right - left), 0.0f, 0.0f, 0.0f};
    m.columns[1] = simd::float4{0.0f, 2.0f / (top - bottom), 0.0f, 0.0f};
    m.columns[2] = simd::float4{0.0f, 0.0f, 1.0f / (zn - zf), 0.0f};
    m.columns[3] = simd::float4{
        -(right + left) / (right - left),
        -(top + bottom) / (top - bottom),
        zn / (zn - zf),
        1.0f
    };
    return m;
}

void ShadowRenderer::CreateResources(id<MTLDevice> device)
{
    MTLTextureDescriptor* shadowDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                           width:kMapSize
                                                          height:kMapSize
                                                       mipmapped:NO];
    shadowDesc.storageMode = MTLStorageModePrivate;
    shadowDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

    for (uint32_t i = 0; i < kCascadeCount; ++i)
    {
        m_maps[i] = [device newTextureWithDescriptor:shadowDesc];
    }

    m_constantBuffer = [device newBufferWithLength:sizeof(ShadowCB)
                                          options:MTLResourceStorageModeShared];
    ResetConstantBuffer();
}

void ShadowRenderer::CreatePipelines(id<MTLDevice> device,
                                     id<MTLLibrary> library,
                                     MTLVertexDescriptor* vertexDescriptor)
{
    NSError* err = nil;

    id<MTLFunction> vsShadow = [library newFunctionWithName:@"vs_shadow"];
    MTLRenderPipelineDescriptor* shadowDesc = [MTLRenderPipelineDescriptor new];
    shadowDesc.vertexFunction = vsShadow;
    shadowDesc.fragmentFunction = nil;
    shadowDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    shadowDesc.vertexDescriptor = vertexDescriptor;

    m_shadowPSO = [device newRenderPipelineStateWithDescriptor:shadowDesc error:&err];
    if (!m_shadowPSO) { NSLog(@"Shadow PSO error: %@", err); }

    id<MTLFunction> vsFenceShadow = [library newFunctionWithName:@"vs_fence_shadow"];
    id<MTLFunction> psFenceShadow = [library newFunctionWithName:@"ps_fence_shadow"];
    MTLRenderPipelineDescriptor* fenceShadowDesc = [MTLRenderPipelineDescriptor new];
    fenceShadowDesc.vertexFunction = vsFenceShadow;
    fenceShadowDesc.fragmentFunction = psFenceShadow;
    fenceShadowDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    fenceShadowDesc.vertexDescriptor = vertexDescriptor;

    m_fenceShadowPSO = [device newRenderPipelineStateWithDescriptor:fenceShadowDesc error:&err];
    if (!m_fenceShadowPSO) { NSLog(@"Fence Shadow PSO error: %@", err); }
}

void ShadowRenderer::CreateSampler(id<MTLDevice> device)
{
    MTLSamplerDescriptor* shadowSmpDesc = [MTLSamplerDescriptor new];
    shadowSmpDesc.minFilter = MTLSamplerMinMagFilterLinear;
    shadowSmpDesc.magFilter = MTLSamplerMinMagFilterLinear;
    shadowSmpDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    shadowSmpDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    shadowSmpDesc.compareFunction = MTLCompareFunctionLessEqual;
    m_sampler = [device newSamplerStateWithDescriptor:shadowSmpDesc];
}

id<MTLTexture> ShadowRenderer::Map(uint32_t cascadeIndex) const
{
    if (cascadeIndex >= kCascadeCount)
    {
        return nil;
    }
    return m_maps[cascadeIndex];
}

void ShadowRenderer::ResetConstantBuffer()
{
    ShadowCB* shadowCb = m_constantBuffer ? (ShadowCB*)m_constantBuffer.contents : nullptr;
    if (!shadowCb)
    {
        return;
    }

    *shadowCb = ShadowCB{};
    shadowCb->params = simd::float4{0.0018f, 0.0065f, 0.58f, m_enabled ? 4.0f : 0.0f};
    for (uint32_t i = 0; i < kCascadeCount; ++i)
    {
        shadowCb->lightViewProj[i] = matrix_identity_float4x4;
        shadowCb->texelSizes[i] = 1.0f / (float)kMapSize;
    }
}

void ShadowRenderer::RenderCascades(id<MTLCommandBuffer> commandBuffer,
                                    id<MTLDepthStencilState> depthStencilState,
                                    id<MTLBuffer> defaultVertexBuffer,
                                    id<MTLSamplerState> vertexSampler,
                                    simd::float3 cameraPosition,
                                    simd::float3 cameraForward,
                                    simd::float3 lightDirection,
                                    float nearPlane,
                                    float farPlane,
                                    float tanHalfFovX,
                                    float tanHalfFovY,
                                    const DrawCascadeCallback& drawCascade)
{
    ResetConstantBuffer();

    ShadowCB* shadowCb = m_constantBuffer ? (ShadowCB*)m_constantBuffer.contents : nullptr;
    if (!m_enabled || !m_shadowPSO || !shadowCb)
    {
        return;
    }

    float cascadeSplits[kCascadeCount] = {};
    constexpr float kCascadeSplitLambda = 0.82f;
    const float safeNearPlane = fmaxf(nearPlane, 0.001f);
    const float safeFarPlane = fmaxf(farPlane, safeNearPlane + 0.001f);
    for (uint32_t i = 0; i < kCascadeCount; ++i)
    {
        const float p = (float)(i + 1u) / (float)kCascadeCount;
        const float logSplit = safeNearPlane * powf(safeFarPlane / safeNearPlane, p);
        const float uniformSplit = safeNearPlane + (safeFarPlane - safeNearPlane) * p;
        cascadeSplits[i] = logSplit * kCascadeSplitLambda + uniformSplit * (1.0f - kCascadeSplitLambda);
    }
    cascadeSplits[kCascadeCount - 1u] = safeFarPlane;
    shadowCb->cascadeSplits =
        simd::float4{cascadeSplits[0], cascadeSplits[1], cascadeSplits[2], cascadeSplits[3]};

    simd::float3 shadowRight = simd::normalize(simd::cross(cameraForward, simd::float3{0.0f, 1.0f, 0.0f}));
    if (simd::length_squared(shadowRight) < 1e-5f)
    {
        shadowRight = simd::float3{1.0f, 0.0f, 0.0f};
    }
    const simd::float3 shadowUp = simd::normalize(simd::cross(shadowRight, cameraForward));
    const simd::float3 lightDir = simd::normalize(lightDirection);

    float cascadeNear = nearPlane;
    for (uint32_t cascadeIndex = 0; cascadeIndex < kCascadeCount; ++cascadeIndex)
    {
        const float cascadeFar = cascadeSplits[cascadeIndex];
        simd::float3 corners[8];
        uint32_t cornerIndex = 0;
        for (float depth : {cascadeNear, cascadeFar})
        {
            const float halfHeight = depth * tanHalfFovY;
            const float halfWidth = depth * tanHalfFovX;
            const simd::float3 center = cameraPosition + cameraForward * depth;
            corners[cornerIndex++] = center - shadowRight * halfWidth - shadowUp * halfHeight;
            corners[cornerIndex++] = center + shadowRight * halfWidth - shadowUp * halfHeight;
            corners[cornerIndex++] = center - shadowRight * halfWidth + shadowUp * halfHeight;
            corners[cornerIndex++] = center + shadowRight * halfWidth + shadowUp * halfHeight;
        }

        simd::float3 cascadeCenter = simd::float3{0.0f, 0.0f, 0.0f};
        for (const simd::float3& corner : corners)
        {
            cascadeCenter += corner;
        }
        cascadeCenter /= 8.0f;

        float cascadeRadius = 0.0f;
        for (const simd::float3& corner : corners)
        {
            cascadeRadius = fmaxf(cascadeRadius, simd::distance(cascadeCenter, corner));
        }
        cascadeRadius = ceilf(cascadeRadius * 16.0f) / 16.0f;

        const simd::float4x4 lightView =
            LookAtRH(cascadeCenter - lightDir * (cascadeRadius * 2.0f),
                     cascadeCenter,
                     simd::float3{0.0f, 1.0f, 0.0f});
        const simd::float4x4 lightProj =
            OrthoRH(-cascadeRadius,
                    cascadeRadius,
                    -cascadeRadius,
                    cascadeRadius,
                    0.1f,
                    cascadeRadius * 4.0f);
        shadowCb->lightViewProj[cascadeIndex] = simd_mul(lightProj, lightView);

        if (!m_maps[cascadeIndex])
        {
            cascadeNear = cascadeFar;
            continue;
        }

        MTLRenderPassDescriptor* shadowPass = [MTLRenderPassDescriptor renderPassDescriptor];
        shadowPass.depthAttachment.texture = m_maps[cascadeIndex];
        shadowPass.depthAttachment.loadAction = MTLLoadActionClear;
        shadowPass.depthAttachment.storeAction = MTLStoreActionStore;
        shadowPass.depthAttachment.clearDepth = 1.0;

        id<MTLRenderCommandEncoder> shadowEnc = [commandBuffer renderCommandEncoderWithDescriptor:shadowPass];
        if (shadowEnc)
        {
            [shadowEnc setRenderPipelineState:m_shadowPSO];
            [shadowEnc setDepthStencilState:depthStencilState];
            [shadowEnc setVertexBuffer:defaultVertexBuffer offset:0 atIndex:0];
            [shadowEnc setVertexSamplerState:vertexSampler atIndex:0];

            if (drawCascade)
            {
                drawCascade(shadowEnc, lightView, lightProj, m_shadowPSO, m_fenceShadowPSO);
            }

            [shadowEnc endEncoding];
        }

        cascadeNear = cascadeFar;
    }
}
