#pragma once

#import <MetalKit/MetalKit.h>
#include <simd/simd.h>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "SceneAssetLoader.hpp"

class Scene
{
public:
    using DrawBatch = SceneDrawBatch;
    using Instance = SceneInstance;
    using CollisionAabb = SceneCollisionAabb;

    using TextureLoader = std::function<id<MTLTexture>(const std::string& path, bool srgb)>;

    bool Load(TextureLoader textureLoader);
    void BuildBvh();
    void UpdateVisibility(const simd::float4x4& viewMatrix,
                          float nearPlane,
                          float farPlane,
                          float tanHalfFovX,
                          float tanHalfFovY,
                          bool enableFrustumCulling,
                          bool enableBvhFrustumCulling);
    void UpdateModel4PlaneStates(simd::float3 cameraPosition,
                                 float enterDistance,
                                 float exitDistance);

    uint32_t IndexCount() const { return m_indexCount; }
    const std::vector<VertexPNT>& Vertices() const { return m_vertices; }
    const std::vector<uint32_t>& Indices() const { return m_indices; }
    const std::vector<DrawBatch>& Batches() const { return m_batches; }
    const std::vector<Instance>& Instances() const { return m_instances; }
    const std::vector<uint8_t>& VisibleInstances() const { return m_visibleInstances; }
    const std::vector<uint8_t>& Model4PlaneStates() const { return m_model4PlaneStates; }
    const std::vector<MaterialGPU>& Materials() const { return m_materials; }
    const std::vector<id<MTLTexture>>& DiffuseTextures() const { return m_diffuseTextures; }
    const std::vector<id<MTLTexture>>& NormalTextures() const { return m_normalTextures; }
    const std::vector<id<MTLTexture>>& HeightTextures() const { return m_heightTextures; }
    const std::vector<CollisionAabb>& CollisionAabbs() const { return m_collisionAabbs; }

    simd::float3 MeshAabbMin() const { return m_meshAabbMin; }
    simd::float3 MeshAabbMax() const { return m_meshAabbMax; }
    simd::float3 MeshCenter() const { return m_meshCenter; }
    float MeshRadius() const { return m_meshRadius; }

    float GetTessellationStrengthForModel(uint32_t modelIndex) const;
    float GetScaleForModel(uint32_t modelIndex) const;

private:
    using ModelBounds = SceneModelBounds;

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

    uint32_t BuildBvhNode(uint32_t begin, uint32_t end);

    uint32_t m_indexCount = 0;
    std::vector<VertexPNT> m_vertices;
    std::vector<uint32_t> m_indices;
    std::vector<DrawBatch> m_batches;
    std::vector<ModelBounds> m_modelBounds;
    std::vector<Instance> m_instances;
    std::vector<uint32_t> m_bvhInstanceIndices;
    std::vector<BvhNode> m_bvhNodes;
    std::vector<uint8_t> m_visibleInstances;
    std::vector<MaterialGPU> m_materials;
    std::vector<id<MTLTexture>> m_diffuseTextures;
    std::vector<id<MTLTexture>> m_normalTextures;
    std::vector<id<MTLTexture>> m_heightTextures;
    std::vector<uint8_t> m_model4PlaneStates;
    std::vector<CollisionAabb> m_collisionAabbs;

    SceneLoadConfig m_loadConfig;
    simd::float3 m_meshAabbMin = {-0.5f, -0.5f, -0.5f};
    simd::float3 m_meshAabbMax = { 0.5f,  0.5f,  0.5f};
    simd::float3 m_meshCenter = {0.0f, 0.0f, 0.0f};
    float m_meshRadius = 1.0f;
    std::vector<float> m_modelTessellationStrengths = {0.0f, 0.0005f, 0.00020f, 0.00020f, 0.0f, 0.0f};
};
