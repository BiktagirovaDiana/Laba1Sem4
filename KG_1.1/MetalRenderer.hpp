#pragma once
#import <MetalKit/MetalKit.h>
#include <cstdint>
#include <string>
#include <vector>
#include "DirectionalLight.hpp"
#include "GBuffer.hpp"
#include "Models.hpp"
#include "Particle.hpp"
#include "Scene.hpp"
#include "ShadowRenderer.hpp"
#include "Terrain.hpp"

class MetalRenderer
{
public:
    explicit MetalRenderer(MTKView* view);
    ~MetalRenderer();

    void DrawFrame();

private:
    struct StructuredBufferElement
    {
        uint32_t value = 0;
        uint32_t pad0 = 0;
        uint32_t pad1 = 0;
        uint32_t pad2 = 0;
    };

    static constexpr uint32_t kStructuredBufferCapacity = 1024;

    MTKView* m_view = nullptr;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;

    id<MTLRenderPipelineState> m_gbufferPSO = nil;
    id<MTLRenderPipelineState> m_lightingPSO = nil;
    id<MTLDepthStencilState>   m_dss = nil;
    id<MTLTexture>             m_whiteTex = nil;
    id<MTLTexture>             m_blackTex = nil;
    id<MTLTexture>             m_flatNormalTex = nil;
    id<MTLTexture>             m_fallbackIrradianceMap = nil;
    id<MTLTexture>             m_irradianceMap = nil;
    id<MTLTexture>             m_fallbackBrdfLut = nil;
    id<MTLTexture>             m_brdfLut = nil;
    id<MTLTexture>             m_fallbackPrefilteredMap = nil;
    id<MTLTexture>             m_prefilteredMap = nil;
    id<MTLSamplerState>        m_sampler = nil;
    GBuffer                    m_gbuffer;

    id<MTLBuffer> m_cameraCB = nil;

    id<MTLBuffer> m_vb = nil;
    id<MTLBuffer> m_ib = nil;
    id<MTLBuffer> m_appendStructuredBuffer = nil;
    id<MTLBuffer> m_appendCounterBuffer = nil;
    id<MTLBuffer> m_consumeStructuredBuffer = nil;
    id<MTLBuffer> m_consumeCounterBuffer = nil;
    Scene m_scene;
    Models m_models;
    ShadowRenderer m_shadowRenderer;
    ParticleRenderer m_particleRenderer;
    Terrain m_terrain;
    simd::float3 m_camPos = { 0.0f, 0.0f, 3.0f };
    float        m_camSpeed = 120.0f; // units/sec
    
    float m_yaw   = 0.0f;
    float m_pitch = 0.0f;
    float m_prevMouseX = 0.0f;
    float m_prevMouseY = 0.0f;
    bool  m_mouseInit  = false;

    float m_mouseSens  = 0.0025f;
    float m_timeSeconds = 0.0f;
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
    bool m_enableVintagePostProcess = false;
    bool m_enableChromaticAberration = false;
    bool m_enableEyeAdaptationPostProcess = false;
    bool m_useBeckmannNDF = false;
    float m_model4PlaneSwapDistance = 200.0f;
    float m_model4PlaneSwapHysteresis = 25.0f;

    void CreateTerrainResources();
    void CreateModelResources();
    void CreateDeviceAndSwapchain();
    void CreateDepth();
    void CreateShadowResources();
    void CreateShadersAndPSO();
    void CreateConstantBuffer();
    void CreateStructuredBuffers();
    void CreateParticleResources();
    void BuildRainCollisionPlanes();
    void LoadObjMesh();
    void CreateSamplerAndFallbackTexture();
    void LoadIrradianceMap();
    void LoadBrdfLut();
    void LoadPrefilteredMap();
    id<MTLTexture> LoadTextureOrNil(const std::string& path, bool srgb);
    id<MTLTexture> LoadCubeTextureOrNil(const std::string& path, bool srgb);
};
