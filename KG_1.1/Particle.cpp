#include "Particle.hpp"

#include <cmath>
#include <random>

namespace
{
constexpr float kPi = 3.14159265359f;
constexpr float kTwoPi = kPi * 2.0f;

simd::float3 RandomUnitVector(std::mt19937& rng)
{
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);

    const float z = unitDist(rng) * 2.0f - 1.0f;
    const float angle = unitDist(rng) * kTwoPi;
    const float xyRadius = sqrtf(fmaxf(0.0f, 1.0f - z * z));

    return simd::float3{cosf(angle) * xyRadius, sinf(angle) * xyRadius, z};
}

simd::float3 RandomPointInCube(std::mt19937& rng, float radius)
{
    std::uniform_real_distribution<float> axisDist(-radius, radius);
    return simd::float3{axisDist(rng), axisDist(rng), axisDist(rng)};
}
}

Particle::Particle(uint32_t planeCount,
                   float radius,
                   float planeSize,
                   simd::float3 center,
                   VolumeShape volumeShape)
    : m_planeCount(planeCount),
      m_radius(radius),
      m_planeSize(planeSize),
      m_center(center),
      m_volumeShape(volumeShape)
{
}

ObjMesh Particle::CreateMesh() const
{
    ObjMesh mesh;
    if (m_planeCount == 0)
    {
        return mesh;
    }

    const float radius = fmaxf(m_radius, 0.0f);
    const float halfSize = fmaxf(m_planeSize, 0.0f) * 0.5f;

    mesh.vertices.reserve((size_t)m_planeCount * 4u);
    mesh.indices.reserve((size_t)m_planeCount * 6u);

    std::mt19937 rng(1337u);
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);

    for (uint32_t i = 0; i < m_planeCount; ++i)
    {
        simd::float3 planeCenter = m_center;
        if (m_volumeShape == VolumeShape::Cube)
        {
            planeCenter += RandomPointInCube(rng, radius);
        }
        else
        {
            const simd::float3 sphereDirection = RandomUnitVector(rng);
            const float distance = cbrtf(unitDist(rng)) * radius;
            planeCenter += sphereDirection * distance;
        }

        const simd::float3 normal = RandomUnitVector(rng);
        const simd::float3 axisHint =
            fabsf(normal.y) < 0.95f
                ? simd::float3{0.0f, 1.0f, 0.0f}
                : simd::float3{1.0f, 0.0f, 0.0f};
        const simd::float3 right = simd::normalize(simd::cross(axisHint, normal));
        const simd::float3 up = simd::normalize(simd::cross(normal, right));

        const simd::float3 p0 = planeCenter - right * halfSize - up * halfSize;
        const simd::float3 p1 = planeCenter + right * halfSize - up * halfSize;
        const simd::float3 p2 = planeCenter + right * halfSize + up * halfSize;
        const simd::float3 p3 = planeCenter - right * halfSize + up * halfSize;

        const uint32_t baseVertex = (uint32_t)mesh.vertices.size();
        mesh.vertices.push_back(VertexPNT{p0.x, p0.y, p0.z, normal.x, normal.y, normal.z, 0.0f, 1.0f});
        mesh.vertices.push_back(VertexPNT{p1.x, p1.y, p1.z, normal.x, normal.y, normal.z, 1.0f, 1.0f});
        mesh.vertices.push_back(VertexPNT{p2.x, p2.y, p2.z, normal.x, normal.y, normal.z, 1.0f, 0.0f});
        mesh.vertices.push_back(VertexPNT{p3.x, p3.y, p3.z, normal.x, normal.y, normal.z, 0.0f, 0.0f});

        mesh.indices.push_back(baseVertex + 0u);
        mesh.indices.push_back(baseVertex + 1u);
        mesh.indices.push_back(baseVertex + 2u);
        mesh.indices.push_back(baseVertex + 0u);
        mesh.indices.push_back(baseVertex + 2u);
        mesh.indices.push_back(baseVertex + 3u);
    }

    ObjMaterial material;
    material.name = "particles";
    material.kd[0] = 0.9f;
    material.kd[1] = 0.95f;
    material.kd[2] = 1.0f;
    material.ks[0] = 0.0f;
    material.ks[1] = 0.0f;
    material.ks[2] = 0.0f;
    material.ns = 8.0f;
    material.d = 1.0f;
    mesh.materials.push_back(material);

    ObjSubmesh submesh;
    submesh.indexOffset = 0;
    submesh.indexCount = (uint32_t)mesh.indices.size();
    submesh.materialIndex = 0;
    mesh.submeshes.push_back(submesh);

    return mesh;
}
