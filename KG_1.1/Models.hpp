#pragma once

#import <MetalKit/MetalKit.h>
#include <simd/simd.h>
#include <functional>
#include <string>
#include "MeshTypes.hpp"
#include "Scene.hpp"

struct CameraCB
{
    simd::float4x4 world;
    simd::float4x4 view;
    simd::float4x4 proj;
    simd::float3   lightDir;
    float          lightIntensity;
    simd::float3   lightColor;
    float          pad0;
    simd::float3   cameraPos;
    float          timeSeconds;
    simd::float4   postProcessParams;
    simd::float4   postProcessParams2;
};

class Models
{
public:
    using AssetResolver = std::function<std::string(const std::string& fileName)>;
    using TextureLoader = std::function<id<MTLTexture>(const std::string& path, bool srgb)>;

    void CreatePipelines(id<MTLDevice> device,
                         id<MTLLibrary> library,
                         MTLVertexDescriptor* vertexDescriptor,
                         MTLPixelFormat depthPixelFormat);
    void CreateResources(id<MTLDevice> device,
                         AssetResolver assetResolver,
                         TextureLoader textureLoader);

    void DrawFenceShadowCaster(id<MTLRenderCommandEncoder> encoder,
                               const CameraCB& camera,
                               const simd::float4x4& lightView,
                               const simd::float4x4& lightProj,
                               id<MTLRenderPipelineState> fenceShadowPSO,
                               id<MTLRenderPipelineState> restorePSO,
                               id<MTLSamplerState> sampler,
                               id<MTLTexture> whiteTexture,
                               id<MTLBuffer> restoreVertexBuffer) const;

    void DrawModel4Planes(id<MTLRenderCommandEncoder> encoder,
                          const CameraCB& camera,
                          const Scene& scene,
                          simd::float3 cameraPosition,
                          id<MTLTexture> blackTexture,
                          id<MTLTexture> whiteTexture,
                          id<MTLTexture> flatNormalTexture,
                          id<MTLBuffer> restoreVertexBuffer) const;

    void DrawFence(id<MTLRenderCommandEncoder> encoder,
                   const CameraCB& camera,
                   id<MTLSamplerState> sampler,
                   id<MTLTexture> whiteTexture) const;

    void DrawFenceShadowDecal(id<MTLRenderCommandEncoder> encoder,
                              const CameraCB& camera,
                              id<MTLSamplerState> sampler) const;

private:
    id<MTLRenderPipelineState> m_fenceGbufferPSO = nil;

    id<MTLBuffer> m_model4PlaneVB = nil;
    id<MTLBuffer> m_model4PlaneIB = nil;
    uint32_t m_model4PlaneIndexCount = 0;
    MaterialGPU m_model4PlaneMaterial;
    id<MTLTexture> m_model4PlaneTexture = nil;
    float m_model4PlaneScale = 7.5f;

    id<MTLBuffer> m_fenceVB = nil;
    id<MTLBuffer> m_fenceIB = nil;
    uint32_t m_fenceIndexCount = 0;
    id<MTLTexture> m_fenceTexture = nil;
    MaterialGPU m_fenceMaterial;

    simd::float3 m_fencePosition = {0.0f, 0.0f, -155.0f};
    simd::float2 m_fenceSize = {400.0f, 240.0f};
    float m_fenceYawRadians = 0.0f;
    float m_fencePitchRadians = 0.0f;
    float m_fenceRollRadians = 0.0f;

    MaterialGPU m_fenceShadowMaterial;
    simd::float3 m_fenceShadowPosition = {0.0f, -5.92f, -131.0f};
    simd::float2 m_fenceShadowSize = {480.0f, 300.0f};
    float m_fenceShadowYawRadians = 0.0f;
    float m_fenceShadowPitchRadians = (float)(M_PI * -0.5);
    float m_fenceShadowRollRadians = 0.0f;
};
