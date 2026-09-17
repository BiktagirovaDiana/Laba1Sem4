#import "Scene.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

static simd::float3 MinVec3(simd::float3 a, simd::float3 b)
{
    return simd::float3{fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
}

static simd::float3 MaxVec3(simd::float3 a, simd::float3 b)
{
    return simd::float3{fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
}

static float ComponentForAxis(simd::float3 v, int axis)
{
    if (axis == 0) return v.x;
    if (axis == 1) return v.y;
    return v.z;
}

static bool IsSphereVisibleInFrustum(const simd::float4x4& viewMatrix,
                                     simd::float3 worldCenter,
                                     float radius,
                                     float nearPlane,
                                     float farPlane,
                                     float tanHalfFovX,
                                     float tanHalfFovY)
{
    const simd::float4 centerView4 = viewMatrix * simd::float4{worldCenter.x, worldCenter.y, worldCenter.z, 1.0f};
    const simd::float3 centerView = simd::float3{centerView4.x, centerView4.y, centerView4.z};
    const float depth = -centerView.z;

    if (depth + radius < nearPlane) return false;
    if (depth - radius > farPlane) return false;
    if (fabsf(centerView.x) > depth * tanHalfFovX + radius) return false;
    if (fabsf(centerView.y) > depth * tanHalfFovY + radius) return false;
    return true;
}

bool Scene::Load(TextureLoader textureLoader)
{
    SceneLoadResult loadedScene;
    SceneAssetLoader loader;
    if (!loader.Load(m_loadConfig, textureLoader, loadedScene))
    {
        m_indexCount = 0;
        return false;
    }

    m_indexCount = loadedScene.indexCount;
    m_vertices = std::move(loadedScene.vertices);
    m_indices = std::move(loadedScene.indices);
    m_batches = std::move(loadedScene.batches);
    m_modelBounds = std::move(loadedScene.modelBounds);
    m_instances = std::move(loadedScene.instances);
    m_model4PlaneStates = std::move(loadedScene.model4PlaneStates);
    m_collisionAabbs = std::move(loadedScene.collisionAabbs);
    m_materials = std::move(loadedScene.materials);
    m_diffuseTextures = std::move(loadedScene.diffuseTextures);
    m_normalTextures = std::move(loadedScene.normalTextures);
    m_heightTextures = std::move(loadedScene.heightTextures);
    m_meshAabbMin = loadedScene.meshAabbMin;
    m_meshAabbMax = loadedScene.meshAabbMax;
    m_meshCenter = loadedScene.meshCenter;
    m_meshRadius = loadedScene.meshRadius;
    m_bvhNodes.clear();
    m_bvhInstanceIndices.clear();
    m_visibleInstances.clear();

    BuildBvh();
    return true;
}

float Scene::GetTessellationStrengthForModel(uint32_t modelIndex) const
{
    if (modelIndex < m_modelTessellationStrengths.size())
    {
        return m_modelTessellationStrengths[modelIndex];
    }
    return m_loadConfig.tessellationStrength;
}

float Scene::GetScaleForModel(uint32_t modelIndex) const
{
    if (modelIndex < m_loadConfig.modelScales.size())
    {
        return m_loadConfig.modelScales[modelIndex];
    }
    return 1.0f;
}

void Scene::UpdateModel4PlaneStates(simd::float3 cameraPosition,
                                    float enterDistance,
                                    float exitDistance)
{
    for (uint32_t instanceIndex = 0; instanceIndex < m_instances.size(); ++instanceIndex)
    {
        const Instance& instance = m_instances[instanceIndex];
        if (instance.sourceModelIndex != 3u)
        {
            continue;
        }

        const float distance = simd::distance(cameraPosition, instance.worldCenter);
        if (instanceIndex >= m_model4PlaneStates.size())
        {
            continue;
        }

        if (!m_model4PlaneStates[instanceIndex] && distance > enterDistance)
        {
            m_model4PlaneStates[instanceIndex] = 1u;
        }
        else if (m_model4PlaneStates[instanceIndex] && distance < exitDistance)
        {
            m_model4PlaneStates[instanceIndex] = 0u;
        }
    }
}

void Scene::UpdateVisibility(const simd::float4x4& viewMatrix,
                             float nearPlane,
                             float farPlane,
                             float tanHalfFovX,
                             float tanHalfFovY,
                             bool enableFrustumCulling,
                             bool enableBvhFrustumCulling)
{
    if (!m_visibleInstances.empty())
    {
        std::fill(m_visibleInstances.begin(), m_visibleInstances.end(), 0u);
    }

    if (!enableFrustumCulling)
    {
        std::fill(m_visibleInstances.begin(), m_visibleInstances.end(), 1u);
        return;
    }

    if (enableBvhFrustumCulling && !m_bvhNodes.empty())
    {
        std::vector<uint32_t> nodeStack;
        nodeStack.push_back(0u);
        while (!nodeStack.empty())
        {
            const uint32_t nodeIndex = nodeStack.back();
            nodeStack.pop_back();

            const BvhNode& node = m_bvhNodes[nodeIndex];
            const simd::float3 nodeCenter = (node.aabbMin + node.aabbMax) * 0.5f;
            const float nodeRadius = simd::length((node.aabbMax - node.aabbMin) * 0.5f);
            if (!IsSphereVisibleInFrustum(viewMatrix,
                                          nodeCenter,
                                          nodeRadius,
                                          nearPlane,
                                          farPlane,
                                          tanHalfFovX,
                                          tanHalfFovY))
            {
                continue;
            }

            if (node.isLeaf)
            {
                for (uint32_t i = 0; i < node.instanceCount; ++i)
                {
                    const uint32_t instanceIndex = m_bvhInstanceIndices[node.firstInstance + i];
                    const Instance& instance = m_instances[instanceIndex];
                    if (IsSphereVisibleInFrustum(viewMatrix,
                                                 instance.worldCenter,
                                                 instance.worldRadius,
                                                 nearPlane,
                                                 farPlane,
                                                 tanHalfFovX,
                                                 tanHalfFovY))
                    {
                        m_visibleInstances[instanceIndex] = 1u;
                    }
                }
            }
            else
            {
                if (node.leftChild != UINT32_MAX)
                {
                    nodeStack.push_back(node.leftChild);
                }
                if (node.rightChild != UINT32_MAX)
                {
                    nodeStack.push_back(node.rightChild);
                }
            }
        }
        return;
    }

    for (uint32_t instanceIndex = 0; instanceIndex < m_instances.size(); ++instanceIndex)
    {
        const Instance& instance = m_instances[instanceIndex];
        if (IsSphereVisibleInFrustum(viewMatrix,
                                     instance.worldCenter,
                                     instance.worldRadius,
                                     nearPlane,
                                     farPlane,
                                     tanHalfFovX,
                                     tanHalfFovY))
        {
            m_visibleInstances[instanceIndex] = 1u;
        }
    }
}

void Scene::BuildBvh()
{
    m_bvhNodes.clear();
    m_bvhInstanceIndices.clear();
    m_visibleInstances.assign(m_instances.size(), 0u);
    if (m_instances.empty())
    {
        return;
    }

    m_bvhInstanceIndices.resize(m_instances.size());
    for (uint32_t i = 0; i < m_instances.size(); ++i)
    {
        m_bvhInstanceIndices[i] = i;
    }

    BuildBvhNode(0u, (uint32_t)m_bvhInstanceIndices.size());
}

uint32_t Scene::BuildBvhNode(uint32_t begin, uint32_t end)
{
    BvhNode node;
    node.firstInstance = begin;
    node.instanceCount = end - begin;

    const Instance& firstInstance = m_instances[m_bvhInstanceIndices[begin]];
    simd::float3 boundsMin = firstInstance.worldAabbMin;
    simd::float3 boundsMax = firstInstance.worldAabbMax;
    simd::float3 centroidMin = firstInstance.worldCenter;
    simd::float3 centroidMax = firstInstance.worldCenter;

    for (uint32_t i = begin + 1; i < end; ++i)
    {
        const Instance& instance = m_instances[m_bvhInstanceIndices[i]];
        boundsMin = MinVec3(boundsMin, instance.worldAabbMin);
        boundsMax = MaxVec3(boundsMax, instance.worldAabbMax);
        centroidMin = MinVec3(centroidMin, instance.worldCenter);
        centroidMax = MaxVec3(centroidMax, instance.worldCenter);
    }

    node.aabbMin = boundsMin;
    node.aabbMax = boundsMax;

    const uint32_t nodeIndex = (uint32_t)m_bvhNodes.size();
    m_bvhNodes.push_back(node);

    const uint32_t instanceCount = end - begin;
    if (instanceCount <= 16u)
    {
        m_bvhNodes[nodeIndex].isLeaf = true;
        return nodeIndex;
    }

    const simd::float3 centroidExtent = centroidMax - centroidMin;
    int splitAxis = 0;
    if (centroidExtent.y > centroidExtent.x && centroidExtent.y >= centroidExtent.z)
    {
        splitAxis = 1;
    }
    else if (centroidExtent.z > centroidExtent.x && centroidExtent.z >= centroidExtent.y)
    {
        splitAxis = 2;
    }

    if (ComponentForAxis(centroidExtent, splitAxis) < 1e-5f)
    {
        m_bvhNodes[nodeIndex].isLeaf = true;
        return nodeIndex;
    }

    const uint32_t mid = begin + instanceCount / 2u;
    std::nth_element(m_bvhInstanceIndices.begin() + begin,
                     m_bvhInstanceIndices.begin() + mid,
                     m_bvhInstanceIndices.begin() + end,
                     [&](uint32_t lhsIndex, uint32_t rhsIndex)
                     {
                         const float lhs = ComponentForAxis(m_instances[lhsIndex].worldCenter, splitAxis);
                         const float rhs = ComponentForAxis(m_instances[rhsIndex].worldCenter, splitAxis);
                         return lhs < rhs;
                     });

    m_bvhNodes[nodeIndex].leftChild = BuildBvhNode(begin, mid);
    m_bvhNodes[nodeIndex].rightChild = BuildBvhNode(mid, end);
    return nodeIndex;
}
