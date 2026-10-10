#pragma once

#import <Metal/Metal.h>
#include <simd/simd.h>

class TAA
{
public:
    struct ResolveParams
    {
        simd::float4x4 previousViewProjectionMatrix;
        simd::float4 renderSizeAndBlend;
        simd::float4 flags;
    };

    TAA();

    void CreatePipelines(id<MTLDevice> device, id<MTLLibrary> library, MTLPixelFormat pixelFormat);
    void EnsureResources(id<MTLDevice> device, NSUInteger width, NSUInteger height, MTLPixelFormat pixelFormat);
    void ResetHistory();

    bool IsReady() const;
    id<MTLTexture> CurrentColorTexture() const;

    MTLRenderPassDescriptor* CreateCurrentColorPassDescriptor() const;

    void EncodeResolve(id<MTLCommandBuffer> commandBuffer,
                       MTLRenderPassDescriptor* drawablePass,
                       id<MTLSamplerState> sampler,
                       id<MTLTexture> albedoTexture,
                       id<MTLTexture> velocityTexture,
                       const ResolveParams& params);

private:
    id<MTLRenderPipelineState> m_resolvePSO = nil;
    id<MTLRenderPipelineState> m_copyPSO = nil;
    id<MTLTexture> m_currentColorTexture = nil;
    id<MTLTexture> m_historyTextures[2] = { nil, nil };
    NSUInteger m_width = 0;
    NSUInteger m_height = 0;
    MTLPixelFormat m_pixelFormat = MTLPixelFormatInvalid;
    uint32_t m_historyIndex = 0;
    bool m_hasHistory = false;
};
