#pragma once

#include <simd/simd.h>

class SpotLight
{
public:
    SpotLight(simd::float3 position,
              simd::float3 direction,
              float range,
              float coneAngleDegrees,
              simd::float3 color,
              float intensity);

    simd::float3 GetPosition() const;
    simd::float3 GetDirection() const;
    float GetRange() const;
    float GetConeAngleRadians() const;
    simd::float3 GetColor() const;
    float GetIntensity() const;

private:
    simd::float3 m_position;
    simd::float3 m_direction;
    float m_range = 1.0f;
    float m_coneAngleRadians = 0.5f;
    simd::float3 m_color;
    float m_intensity = 1.0f;
};
