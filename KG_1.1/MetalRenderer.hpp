#pragma once
#import <MetalKit/MetalKit.h>
#include <vector>
#include "DirectionalLight.hpp"
#include "GBuffer.hpp"
#include "ObjLoader.hpp"
#include "PointLight.hpp"
#include "SpotLight.hpp"

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
    };

    struct MaterialGPU
    {
        simd::float4 kd_ns = {1.0f, 1.0f, 1.0f, 32.0f};
        simd::float4 ks_alpha = {0.0f, 0.0f, 0.0f, 1.0f};
        simd::float2 uvScale = {1.0f, 1.0f};
        simd::float2 uvSpeed = {0.0f, 0.0f};
        uint32_t useTexture = 0;
        simd::float3 pad = {0.0f, 0.0f, 0.0f};
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

    MTKView* m_view = nullptr;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;

    id<MTLRenderPipelineState> m_gbufferPSO = nil;
    id<MTLRenderPipelineState> m_lightingPSO = nil;
    id<MTLDepthStencilState>   m_dss = nil;
    id<MTLTexture>             m_whiteTex = nil;
    id<MTLSamplerState>        m_sampler = nil;
    GBuffer                    m_gbuffer;

    id<MTLBuffer> m_cameraCB = nil;

    id<MTLBuffer> m_vb = nil;
    id<MTLBuffer> m_ib = nil;
    uint32_t m_indexCount = 0;
    std::vector<VertexPNT> m_cpuVertices;
    std::vector<uint32_t> m_cpuIndices;
    std::vector<CollisionTriangle> m_collisionTriangles;
    std::vector<DrawBatch> m_batches;
    std::vector<MaterialGPU> m_materials;
    std::vector<id<MTLTexture>> m_materialTextures;
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
    DirectionalLight m_directionalLight = DirectionalLight(simd::float3{0.0f, -1.0f, 0.0f},
                                                           simd::float3{1.0f, 1.0f, 1.0f},
                                                           1.0f);
    PointLight m_pointLight = PointLight(simd::float3{0.0f, 0.0f, 0.0f},
                                         1.0f,
                                         simd::float3{1.0f, 1.0f, 1.0f},
                                         1.0f);
    SpotLight m_spotLight = SpotLight(simd::float3{0.0f, 0.0f, 0.0f},
                                      simd::float3{0.0f, -1.0f, 0.0f},
                                      1.0f,
                                      30.0f,
                                      simd::float3{1.0f, 1.0f, 1.0f},
                                      1.0f);
    simd::float3 m_thrownLightPos = {0.0f, 200.0f, 0.0f};
    simd::float3 m_thrownLightVelocity = {0.0f, 0.0f, 0.0f};
    simd::float3 m_thrownLightForward = {0.0f, 0.0f, -1.0f};
    bool m_thrownLightActive = false;
    bool m_thrownLightLanded = false;
    float m_thrownLightGravity = 28.0f;
    float m_thrownLightLaunchSpeed = 220.0f;
    float m_thrownLightSpawnOffset = 12.0f;
    float m_thrownLightInitialDropSpeed = 4.0f;
    float m_thrownLightSurfaceOffset = 3.0f;
    float m_textureAnimTimeSeconds = 0.0f;
    float m_textureNearSpeedMultiplier = 4.0f;
    float m_textureFarSpeedMultiplier = 0.35f;
    float m_textureFarDistance = 8.0f;

    void CreateDeviceAndSwapchain();
    void CreateDepth();
    void CreateShadersAndPSO();
    void CreateConstantBuffer();
    void LoadObjMesh();
    void CreateSamplerAndFallbackTexture();
    id<MTLTexture> LoadTextureOrNil(const std::string& path);
};
