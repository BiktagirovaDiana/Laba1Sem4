#pragma once
#import <MetalKit/MetalKit.h>
#include <cstdint>
#include <vector>
#include "DirectionalLight.hpp"
#include "GBuffer.hpp"
#include "ObjLoader.hpp"
#include "Particle.hpp"

class MetalRenderer
{
public:
    explicit MetalRenderer(MTKView* view);
    ~MetalRenderer();

    void DrawFrame();

private:
    struct DrawBatch
    {
        uint32_t indexOffset = 0;
        uint32_t indexCount = 0;
        uint32_t materialIndex = 0;
        uint32_t sourceModelIndex = 0;
        uint32_t instanceIndex = 0;
    };

    struct ModelBounds
    {
        simd::float3 localAabbMin = {0.0f, 0.0f, 0.0f};
        simd::float3 localAabbMax = {0.0f, 0.0f, 0.0f};
        simd::float3 localCenter = {0.0f, 0.0f, 0.0f};
        float localRadius = 1.0f;
    };

    struct SceneInstance
    {
        uint32_t sourceModelIndex = 0;
        simd::float3 worldOffset = {0.0f, 0.0f, 0.0f};
        simd::float3 worldAabbMin = {0.0f, 0.0f, 0.0f};
        simd::float3 worldAabbMax = {0.0f, 0.0f, 0.0f};
        simd::float3 worldCenter = {0.0f, 0.0f, 0.0f};
        float worldRadius = 1.0f;
        float scale = 1.0f;
    };

    struct BvhNode
    {
        simd::float3 aabbMin = {0.0f, 0.0f, 0.0f};
        simd::float3 aabbMax = {0.0f, 0.0f, 0.0f};
        uint32_t leftChild = UINT32_MAX;
        uint32_t rightChild = UINT32_MAX;
        uint32_t firstInstance = 0;
        uint32_t instanceCount = 0;
        bool isLeaf = false;
    };

    struct MaterialGPU
    {
        simd::float4 kd_ns = {1.0f, 1.0f, 1.0f, 32.0f};
        simd::float4 ks_alpha = {0.0f, 0.0f, 0.0f, 1.0f};
        simd::float2 uvScale = {1.0f, 1.0f};
        simd::float2 uvSpeed = {0.0f, 0.0f};
        simd::uint4 textureFlags = {0u, 0u, 0u, 0u};
        simd::float4 detailParams = {2.0f, 1.0f, 0.0f, 0.0f};
    };

    struct CollisionTriangle
    {
        simd::float3 a;
        simd::float3 b;
        simd::float3 c;
        simd::float3 normal;
        simd::float3 aabbMin;
        simd::float3 aabbMax;
    };

    struct StructuredBufferElement
    {
        uint32_t value = 0;
        uint32_t pad0 = 0;
        uint32_t pad1 = 0;
        uint32_t pad2 = 0;
    };

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

    struct ParticleInstanceGPU
    {
        simd::float4 baseCenterAndSize = {0.0f, 0.0f, 0.0f, 1.0f};
        simd::float4 animatedCenterAndSeed = {0.0f, 0.0f, 0.0f, 0.0f};
    };

    static constexpr uint32_t kStructuredBufferCapacity = 1024;

    MTKView* m_view = nullptr;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;

    id<MTLRenderPipelineState> m_gbufferPSO = nil;
    id<MTLRenderPipelineState> m_particleBillboardPSO = nil;
    id<MTLRenderPipelineState> m_lightingPSO = nil;
    id<MTLComputePipelineState> m_particleComputePSO = nil;
    id<MTLComputePipelineState> m_dustComputePSO = nil;
    id<MTLComputePipelineState> m_rainComputePSO = nil;
    id<MTLDepthStencilState>   m_dss = nil;
    id<MTLTexture>             m_whiteTex = nil;
    id<MTLTexture>             m_blackTex = nil;
    id<MTLTexture>             m_flatNormalTex = nil;
    id<MTLSamplerState>        m_sampler = nil;
    GBuffer                    m_gbuffer;

    id<MTLBuffer> m_cameraCB = nil;

    id<MTLBuffer> m_vb = nil;
    id<MTLBuffer> m_ib = nil;
    id<MTLBuffer> m_model4PlaneVB = nil;
    id<MTLBuffer> m_model4PlaneIB = nil;
    id<MTLBuffer> m_particleQuadVB = nil;
    id<MTLBuffer> m_particleQuadIB = nil;
    id<MTLBuffer> m_particleBaseInstanceBuffer = nil;
    id<MTLBuffer> m_particleInstanceBuffer = nil;
    id<MTLBuffer> m_dustBaseInstanceBuffer = nil;
    id<MTLBuffer> m_dustInstanceBuffer = nil;
    id<MTLBuffer> m_rainBaseInstanceBuffer = nil;
    id<MTLBuffer> m_rainInstanceBuffer = nil;
    id<MTLBuffer> m_rainCollisionPlaneBuffer = nil;
    id<MTLBuffer> m_appendStructuredBuffer = nil;
    id<MTLBuffer> m_appendCounterBuffer = nil;
    id<MTLBuffer> m_consumeStructuredBuffer = nil;
    id<MTLBuffer> m_consumeCounterBuffer = nil;
    uint32_t m_indexCount = 0;
    uint32_t m_model4PlaneIndexCount = 0;
    uint32_t m_particleQuadIndexCount = 0;
    uint32_t m_particleInstanceCount = 0;
    uint32_t m_dustParticleInstanceCount = 0;
    uint32_t m_rainParticleInstanceCount = 0;
    uint32_t m_rainCollisionPlaneCount = 0;
    std::vector<VertexPNT> m_cpuVertices;
    std::vector<uint32_t> m_cpuIndices;
    std::vector<CollisionTriangle> m_collisionTriangles;
    std::vector<DrawBatch> m_batches;
    std::vector<ModelBounds> m_modelBounds;
    std::vector<SceneInstance> m_sceneInstances;
    std::vector<uint32_t> m_bvhInstanceIndices;
    std::vector<BvhNode> m_bvhNodes;
    std::vector<uint8_t> m_visibleInstances;
    std::vector<MaterialGPU> m_materials;
    std::vector<id<MTLTexture>> m_diffuseTextures;
    std::vector<id<MTLTexture>> m_normalTextures;
    std::vector<id<MTLTexture>> m_heightTextures;
    std::vector<uint8_t> m_model4PlaneStates;
    std::vector<VertexPNT> m_particleBaseVertices;
    MaterialGPU m_model4PlaneMaterial;
    id<MTLTexture> m_model4PlaneTexture = nil;
    id<MTLTexture> m_particleTexture = nil;
    id<MTLTexture> m_dustParticleTexture = nil;
    id<MTLTexture> m_rainParticleTexture = nil;
    simd::float3 m_camPos = { 0.0f, 0.0f, 3.0f };
    float        m_camSpeed = 120.0f; // units/sec
    
    float m_yaw   = 0.0f;
    float m_pitch = 0.0f;
    float m_prevMouseX = 0.0f;
    float m_prevMouseY = 0.0f;
    bool  m_mouseInit  = false;

    float m_mouseSens  = 0.0025f; 
    float m_timeSeconds = 0.0f;
    simd::float2 m_textureTiling = {2.0f, 2.0f};
    simd::float2 m_textureScrollSpeed = {0.08f, 0.0f};
    simd::float3 m_meshAabbMin = {-0.5f, -0.5f, -0.5f};
    simd::float3 m_meshAabbMax = { 0.5f,  0.5f,  0.5f};
    simd::float3 m_meshCenter = {0.0f, 0.0f, 0.0f};
    float m_meshRadius = 1.0f;
    float m_tessellationStrength = 0.0005;
    float m_tessellationFadeNearMultiplier = 1.5f;
    float m_tessellationFadeFarMultiplier = 4.5f;
    DirectionalLight m_directionalLight = DirectionalLight(simd::float3{0.0f, -1.0f, 0.0f},
                                                           simd::float3{1.0f, 1.0f, 1.0f},
                                                           1.0f);
    float m_textureAnimTimeSeconds = 0.0f;
    float m_textureNearSpeedMultiplier = 4.0f;
    float m_textureFarSpeedMultiplier = 0.35f;
    float m_textureFarDistance = 8.0f;
    bool m_enableFrustumCulling = true;
    bool m_enableBvhFrustumCulling = true;
    int m_model4InstanceCount = 20000;
    float m_model4PlaneSwapDistance = 200.0f;
    float m_model4PlaneSwapHysteresis = 25.0f;
    float m_model4PlaneScale = 7.5f;
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
    std::vector<float> m_modelTessellationStrengths = {0.0f, 0.0005f, 0.00020f, 0.00020f};
    std::vector<simd::float3> m_modelOffsets =
    {
        simd::float3{0.0f, 0.0f, 0.0f},
        simd::float3{0.0f, 0.0f, 0.0f},
        simd::float3{25.0f, 0.0f, 0.0f},
        simd::float3{50.0f, 0.0f, 0.0f}
    };

    void CreateDeviceAndSwapchain();
    void CreateDepth();
    void CreateShadersAndPSO();
    void CreateConstantBuffer();
    void CreateStructuredBuffers();
    void CreateParticleResources();
    void CreateDustParticleResources();
    void CreateRainParticleResources();
    void BuildRainCollisionPlanes();
    void UpdateParticleAnimation(id<MTLCommandBuffer> commandBuffer, float dt);
    void UpdateDustParticleAnimation(id<MTLCommandBuffer> commandBuffer, float dt);
    void UpdateRainParticleAnimation(id<MTLCommandBuffer> commandBuffer, float dt);
    void LoadObjMesh();
    void CreateModel4PlaneResources();
    void CreateSamplerAndFallbackTexture();
    id<MTLTexture> LoadTextureOrNil(const std::string& path, bool srgb);
    void BuildSceneBVH();
    uint32_t BuildSceneBVHNode(uint32_t begin, uint32_t end);
    float GetTessellationStrengthForModel(uint32_t modelIndex) const;
    simd::float3 GetOffsetForModel(uint32_t modelIndex) const;
};
