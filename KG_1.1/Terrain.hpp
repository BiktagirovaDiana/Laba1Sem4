#pragma once

#import <MetalKit/MetalKit.h>
#include <simd/simd.h>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "MeshTypes.hpp"

class Terrain
{
public:
    struct DrawBatch
    {
        uint32_t indexOffset = 0; //откуда рисуем
        uint32_t indexCount = 0; //сколько индексов рисовать
        uint32_t sourceTileIndex = 0; //какие текстуры использовать
    };

    struct SourceTile //данные исходных тайлов (по одному)
    {
        //данные из карты высоты
        std::vector<float> heights; //высота в диапозоне от 0 до 1
        uint32_t width = 0;
        uint32_t height = 0;
        
        id<MTLTexture> diffuseTexture = nil;
        id<MTLTexture> normalTexture = nil;
    };

    using TextureLoader = std::function<id<MTLTexture>(const std::string& path, bool srgb)>;

    void CreateResources(id<MTLDevice> device, TextureLoader textureLoader);
    void UpdateMesh(const simd::float4x4& viewMatrix,
                    float nearPlane,
                    float farPlane,
                    float tanHalfFovX,
                    float tanHalfFovY,
                    simd::float3 cameraPosition,
                    bool enableFrustumCulling);

    bool IsEnabled() const { return m_enabled; }
    bool IsDrawable() const { return m_enabled && m_vertexBuffer && m_indexBuffer && m_indexCount > 0u; }
    float HalfSize() const { return m_halfSize; }

    id<MTLBuffer> VertexBuffer() const { return m_vertexBuffer; }
    id<MTLBuffer> IndexBuffer() const { return m_indexBuffer; }
    uint32_t IndexCount() const { return m_indexCount; }
    const MaterialGPU& Material() const { return m_material; }
    const std::vector<DrawBatch>& Batches() const { return m_batches; }
    const SourceTile* SourceTileAt(uint32_t index) const;

private:
    struct Tile //временный участок террейна
    {
        simd::float2 center = {0.0f, 0.0f};
        float size = 1.0f;
        uint32_t depth = 0; //глубина разбиения, детализация
    };

    void LoadTiles();
    SourceTile LoadSourceTile(const std::string& heightPath,
                              const std::string& diffusePath,
                              const std::string& normalPath);
    uint32_t GetSourceTileIndex(float x, float z) const;
    float SampleHeightFromTile(const SourceTile& tile, float u, float v) const;
    float SampleHeight(float x, float z) const;
    simd::float3 SampleNormal(float x, float z) const;
    void SelectTilesRecursive(simd::float2 center,
                              float size,
                              uint32_t depth,
                              const simd::float4x4& viewMatrix,
                              float nearPlane,
                              float farPlane,
                              float tanHalfFovX,
                              float tanHalfFovY,
                              simd::float3 cameraPosition,
                              bool enableFrustumCulling,
                              std::vector<Tile>& outTiles) const;
    bool IsTileVisibleInFrustum(simd::float2 center,
                                float size,
                                const simd::float4x4& viewMatrix,
                                float nearPlane,
                                float farPlane,
                                float tanHalfFovX,
                                float tanHalfFovY) const;

    id<MTLDevice> m_device = nil;
    TextureLoader m_textureLoader;

    id<MTLBuffer> m_vertexBuffer = nil;
    id<MTLBuffer> m_indexBuffer = nil;
    uint32_t m_indexCount = 0;
    uint32_t m_tileCount = 0;
    MaterialGPU m_material;
    std::vector<SourceTile> m_sourceTiles;
    std::vector<DrawBatch> m_batches;

    bool m_enabled = true;
    float m_halfSize = 480.0f;
    float m_baseY = 100.0f;
    float m_maxHeight = 42.0f;
    uint32_t m_maxDepth = 6;
    uint32_t m_patchResolution = 8;
    float m_lodDistanceFactor = 1.75f;
};
