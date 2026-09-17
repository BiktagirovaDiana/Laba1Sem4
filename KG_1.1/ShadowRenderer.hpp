#pragma once

#import <MetalKit/MetalKit.h>
#include <simd/simd.h>
#include <cstdint>
#include <functional>

struct ShadowCB
{
    simd::float4x4 lightViewProj[4];
    simd::float4 cascadeSplits = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 texelSizes = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 params = {0.0025f, 0.02f, 0.62f, 4.0f};
};

class ShadowRenderer
{
public:
    static constexpr uint32_t kCascadeCount = 4;
    static constexpr uint32_t kMapSize = 2048;

    using DrawCascadeCallback =
        std::function<void(id<MTLRenderCommandEncoder> encoder,
                           const simd::float4x4& lightView,
                           const simd::float4x4& lightProj,
                           id<MTLRenderPipelineState> shadowPSO,
                           id<MTLRenderPipelineState> fenceShadowPSO)>;

    void CreateResources(id<MTLDevice> device);
    void CreatePipelines(id<MTLDevice> device,
                         id<MTLLibrary> library,
                         MTLVertexDescriptor* vertexDescriptor);
    void CreateSampler(id<MTLDevice> device);

    void RenderCascades(id<MTLCommandBuffer> commandBuffer,
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
                        const DrawCascadeCallback& drawCascade);

    bool IsEnabled() const { return m_enabled; }
    void SetEnabled(bool enabled) { m_enabled = enabled; }

    id<MTLBuffer> ConstantBuffer() const { return m_constantBuffer; }
    id<MTLSamplerState> Sampler() const { return m_sampler; }
    id<MTLTexture> Map(uint32_t cascadeIndex) const;

private:
    void ResetConstantBuffer();

    bool m_enabled = true;
    id<MTLTexture> m_maps[kCascadeCount] = {nil, nil, nil, nil};
    id<MTLBuffer> m_constantBuffer = nil;
    id<MTLSamplerState> m_sampler = nil;
    id<MTLRenderPipelineState> m_shadowPSO = nil;
    id<MTLRenderPipelineState> m_fenceShadowPSO = nil;
};
