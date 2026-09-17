#pragma once

#import <MetalKit/MetalKit.h>
#include <simd/simd.h>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "MeshTypes.hpp"

struct SceneDrawBatch
{
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    uint32_t materialIndex = 0;
    uint32_t sourceModelIndex = 0;
    uint32_t instanceIndex = 0;
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

struct SceneCollisionAabb
{
    simd::float3 min = {0.0f, 0.0f, 0.0f};
    simd::float3 max = {0.0f, 0.0f, 0.0f};
};

struct SceneModelBounds
{
    simd::float3 localAabbMin = {0.0f, 0.0f, 0.0f};
    simd::float3 localAabbMax = {0.0f, 0.0f, 0.0f};
    simd::float3 localCenter = {0.0f, 0.0f, 0.0f};
    float localRadius = 1.0f;
};

struct SceneLoadConfig
{
    simd::float2 textureTiling = {2.0f, 2.0f};
    simd::float2 textureScrollSpeed = {0.08f, 0.0f};
    float tessellationStrength = 0.0005f;
    int model4InstanceCount = 20000;
    std::vector<float> modelScales = {1.0f, 1.0f, 1.0f, 5.0f, 35.0f, 25.0f};
    std::vector<simd::float3> modelOffsets =
    {
        simd::float3{0.0f, 0.0f, 0.0f},
        simd::float3{0.0f, 0.0f, 0.0f},
        simd::float3{25.0f, 0.0f, 0.0f},
        simd::float3{50.0f, 0.0f, 0.0f},
        simd::float3{80.0f, 2.0f, -120.0f},
        simd::float3{125.0f, 12.0f, -100.0f}
    };
};

struct SceneLoadResult
{
    uint32_t indexCount = 0;
    std::vector<VertexPNT> vertices;
    std::vector<uint32_t> indices;
    std::vector<SceneDrawBatch> batches;
    std::vector<SceneModelBounds> modelBounds;
    std::vector<SceneInstance> instances;
    std::vector<uint8_t> model4PlaneStates;
    std::vector<SceneCollisionAabb> collisionAabbs;
    std::vector<MaterialGPU> materials;
    std::vector<id<MTLTexture>> diffuseTextures;
    std::vector<id<MTLTexture>> normalTextures;
    std::vector<id<MTLTexture>> heightTextures;
    simd::float3 meshAabbMin = {-0.5f, -0.5f, -0.5f};
    simd::float3 meshAabbMax = { 0.5f,  0.5f,  0.5f};
    simd::float3 meshCenter = {0.0f, 0.0f, 0.0f};
    float meshRadius = 1.0f;
};

class SceneAssetLoader
{
public:
    using TextureLoader = std::function<id<MTLTexture>(const std::string& path, bool srgb)>;

    bool Load(const SceneLoadConfig& config,
              TextureLoader textureLoader,
              SceneLoadResult& outScene) const;

private:
    float GetScaleForModel(const SceneLoadConfig& config, uint32_t modelIndex) const;
    simd::float3 GetOffsetForModel(const SceneLoadConfig& config, uint32_t modelIndex) const;
};
