#pragma once
#import <MetalKit/MetalKit.h>
#include <vector>
#include "ObjLoader.hpp"

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

    MTKView* m_view = nullptr;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;

    id<MTLRenderPipelineState> m_pso = nil;
    id<MTLRenderPipelineState> m_skyPSO = nil;
    id<MTLDepthStencilState>   m_dss = nil;
    id<MTLTexture>             m_whiteTex = nil;
    id<MTLSamplerState>        m_sampler = nil;

    id<MTLBuffer> m_cameraCB = nil;

    id<MTLBuffer> m_vb = nil;
    id<MTLBuffer> m_ib = nil;
    uint32_t m_indexCount = 0;
    std::vector<DrawBatch> m_batches;
    std::vector<MaterialGPU> m_materials;
    std::vector<id<MTLTexture>> m_materialTextures;
    simd::float3 m_camPos = { 0.0f, 0.0f, 3.0f };
    float        m_camSpeed = 2.5f; // units/sec
    
    float m_yaw   = 0.0f;
    float m_pitch = 0.0f;
    float m_prevMouseX = 0.0f;
    float m_prevMouseY = 0.0f;
    bool  m_mouseInit  = false;

    float m_mouseSens  = 0.0025f; 
    float m_timeSeconds = 0.0f;
    simd::float2 m_textureTiling = {2.0f, 2.0f};
    simd::float2 m_textureScrollSpeed = {0.08f, 0.0f};

    void CreateDeviceAndSwapchain();
    void CreateDepth();
    void CreateShadersAndPSO();
    void CreateConstantBuffer();
    void LoadObjMesh();
    void CreateSamplerAndFallbackTexture();
    id<MTLTexture> LoadTextureOrNil(const std::string& path);
};
