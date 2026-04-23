#pragma once

#include <cstdint>
#include <simd/simd.h>
#include "ObjLoader.hpp"

class Particle
{
public:
    Particle(uint32_t planeCount,
             float radius,
             float planeSize,
             simd::float3 center = simd::float3{0.0f, 0.0f, 0.0f});

    ObjMesh CreateMesh() const;

private:
    uint32_t m_planeCount = 0;
    float m_radius = 1.0f;
    float m_planeSize = 1.0f;
    simd::float3 m_center = {0.0f, 0.0f, 0.0f};
};
