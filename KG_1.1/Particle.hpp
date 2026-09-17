#pragma once

#include <cstdint>
#include <simd/simd.h>
#include "MeshTypes.hpp"

#ifdef __OBJC__
#import <MetalKit/MetalKit.h>
#include <functional>
#include <string>
#include <vector>
#include "Terrain.hpp"
#endif

class Particle
{
public:
    enum class VolumeShape
    {
        Sphere,
        Cube
    };

    Particle(uint32_t planeCount,
             float radius,
             float planeSize,
             simd::float3 center = simd::float3{0.0f, 0.0f, 0.0f},
             VolumeShape volumeShape = VolumeShape::Sphere);

    ObjMesh CreateMesh() const;

private:
    uint32_t m_planeCount = 0;
    float m_radius = 1.0f;
    float m_planeSize = 1.0f;
    simd::float3 m_center = {0.0f, 0.0f, 0.0f};
    VolumeShape m_volumeShape = VolumeShape::Sphere;
};

#ifdef __OBJC__
class ParticleRenderer
{
public:
    struct CollisionAabb
    {
        simd::float3 min = {0.0f, 0.0f, 0.0f};
        simd::float3 max = {0.0f, 0.0f, 0.0f};
    };

    using AssetResolver = std::function<std::string(const std::string& fileName)>;
    using TextureLoader = std::function<id<MTLTexture>(const std::string& path, bool srgb)>;

    struct ParticleInstanceGPU
    {
        simd::float4 baseCenterAndSize = {0.0f, 0.0f, 0.0f, 1.0f};
        simd::float4 animatedCenterAndSeed = {0.0f, 0.0f, 0.0f, 0.0f};
    };

    void CreatePipelines(id<MTLDevice> device,
                         id<MTLLibrary> library,
                         MTLRenderPipelineDescriptor* gbufferPipelineDescriptor);
    void CreateResources(id<MTLDevice> device,
                         AssetResolver assetResolver,
                         TextureLoader textureLoader);
    void BuildRainCollisionPlanes(id<MTLDevice> device,
                                  const std::vector<CollisionAabb>& collisionAabbs);
    void Update(id<MTLCommandBuffer> commandBuffer, float dt);
    void Draw(id<MTLRenderCommandEncoder> encoder,
              const void* cameraConstantBuffer,
              NSUInteger cameraConstantBufferLength,
              id<MTLTexture> blackTexture,
              id<MTLTexture> whiteTexture,
              id<MTLTexture> flatNormalTexture,
              id<MTLBuffer> defaultVertexBuffer);

private:
    struct ParticleAnimationCB
    {
        simd::float4 centerAndFactor = {0.0f, 0.0f, 0.0f, 1.0f};
        uint32_t instanceCount = 0;
        uint32_t pad0 = 0;
        uint32_t pad1 = 0;
        uint32_t pad2 = 0;
    };

    struct DustAnimationCB
    {
        simd::float4 centerAndTime = {0.0f, 0.0f, 0.0f, 0.0f};
        simd::float4 motionParams = {1.0f, 1.0f, 1.0f, 0.0f};
        uint32_t instanceCount = 0;
        uint32_t pad0 = 0;
        uint32_t pad1 = 0;
        uint32_t pad2 = 0;
    };

    struct RainAnimationCB
    {
        simd::float4 centerAndTime = {0.0f, 0.0f, 0.0f, 0.0f};
        simd::float4 volumeAndSpeed = {1.0f, 1.0f, 0.0f, 0.0f};
        simd::float4 bounceParams = {1.0f, 1.0f, 0.0f, 0.0f};
        uint32_t instanceCount = 0;
        uint32_t collisionPlaneCount = 0;
        uint32_t pad0 = 0;
        uint32_t pad1 = 0;
    };

    struct RainCollisionPlaneGPU
    {
        simd::float4 minXZMaxXZ = {0.0f, 0.0f, 0.0f, 0.0f};
        simd::float4 yAndPadding = {0.0f, 0.0f, 0.0f, 0.0f};
    };

    void CreateParticleResources(id<MTLDevice> device,
                                 const std::string& particleTexturePath,
                                 TextureLoader textureLoader);
    void CreateDustParticleResources(id<MTLDevice> device,
                                     const std::string& dustTexturePath,
                                     TextureLoader textureLoader);
    void CreateRainParticleResources(id<MTLDevice> device,
                                     const std::string& rainTexturePath,
                                     TextureLoader textureLoader);
    void UpdateParticleAnimation(id<MTLCommandBuffer> commandBuffer, float dt);
    void UpdateDustParticleAnimation(id<MTLCommandBuffer> commandBuffer, float dt);
    void UpdateRainParticleAnimation(id<MTLCommandBuffer> commandBuffer, float dt);
    void DrawParticleSet(id<MTLRenderCommandEncoder> encoder,
                         const void* cameraConstantBuffer,
                         NSUInteger cameraConstantBufferLength,
                         const MaterialGPU& material,
                         id<MTLBuffer> instanceBuffer,
                         id<MTLTexture> texture,
                         uint32_t instanceCount,
                         id<MTLTexture> blackTexture,
                         id<MTLTexture> whiteTexture,
                         id<MTLTexture> flatNormalTexture);

    id<MTLRenderPipelineState> m_billboardPSO = nil;
    id<MTLComputePipelineState> m_particleComputePSO = nil;
    id<MTLComputePipelineState> m_dustComputePSO = nil;
    id<MTLComputePipelineState> m_rainComputePSO = nil;

    id<MTLBuffer> m_quadVB = nil;
    id<MTLBuffer> m_quadIB = nil;
    id<MTLBuffer> m_particleBaseInstanceBuffer = nil;
    id<MTLBuffer> m_particleInstanceBuffer = nil;
    id<MTLBuffer> m_dustBaseInstanceBuffer = nil;
    id<MTLBuffer> m_dustInstanceBuffer = nil;
    id<MTLBuffer> m_rainBaseInstanceBuffer = nil;
    id<MTLBuffer> m_rainInstanceBuffer = nil;
    id<MTLBuffer> m_rainCollisionPlaneBuffer = nil;

    uint32_t m_quadIndexCount = 0;
    uint32_t m_particleInstanceCount = 0;
    uint32_t m_dustParticleInstanceCount = 0;
    uint32_t m_rainParticleInstanceCount = 0;
    uint32_t m_rainCollisionPlaneCount = 0;

    id<MTLTexture> m_particleTexture = nil;
    id<MTLTexture> m_dustParticleTexture = nil;
    id<MTLTexture> m_rainParticleTexture = nil;

    uint32_t m_particlePlaneCount = 128;
    float m_particleRadius = 35.0f;
    float m_particlePlaneSize = 2.0f;
    simd::float3 m_particleCenter = {0.0f, 2.0f, 0.0f};
    float m_particleAnimationTime = 0.0f;
    float m_particleCycleSeconds = 2.4f;
    float m_particleCollapsePart = 0.72f;
    MaterialGPU m_particleMaterial;

    uint32_t m_dustParticlePlaneCount = 1080;
    float m_dustParticleRadius = 80.0f;
    float m_dustParticlePlaneSize = 0.65f;
    simd::float3 m_dustParticleCenter = {0.0f, 18.0f, 0.0f};
    float m_dustAnimationTime = 0.0f;
    float m_dustDriftAmplitude = 11.0f;
    float m_dustDriftSpeed = 0.22f;
    float m_dustSwirlAmplitude = 2.5f;
    MaterialGPU m_dustParticleMaterial;

    uint32_t m_rainParticlePlaneCount = 1400;
    float m_rainParticleRadius = 75.0f;
    float m_rainParticlePlaneSize = 0.4f;
    simd::float3 m_rainParticleCenter = {0.0f, 70.0f, 0.0f};
    float m_rainAnimationTime = 0.0f;
    float m_rainFallHeight = 150.0f;
    float m_rainFallSpeed = 42.0f;
    float m_rainBounceHeight = 2.4f;
    float m_rainBounceDistance = 6.5f;
    float m_rainCollisionBias = 5.75f;
    MaterialGPU m_rainParticleMaterial;
};
#endif
