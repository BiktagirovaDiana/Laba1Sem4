#pragma once

#include <simd/simd.h>

class PointLight
{

public:
    PointLight(simd::float3 position,
               float range,
               simd::float3 color,
               float intensity);

    simd::float3 GetPosition() const;
    float GetRange() const;
    simd::float3 GetColor() const;
    float GetIntensity() const;

private:
    simd::float3 m_position;
    float m_range = 1.0f;
    simd::float3 m_color;
    float m_intensity = 1.0f;
};
