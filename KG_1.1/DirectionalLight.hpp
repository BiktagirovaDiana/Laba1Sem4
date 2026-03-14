#pragma once

#include <simd/simd.h>

class DirectionalLight
{
public:
    DirectionalLight(simd::float3 direction,
                     simd::float3 color,
                     float intensity);

    simd::float3 GetDirection() const;
    simd::float3 GetColor() const;
    float GetIntensity() const;

private:
    simd::float3 m_direction;
    simd::float3 m_color;
    float m_intensity = 1.0f;
};
