#include "DirectionalLight.hpp"

DirectionalLight::DirectionalLight(simd::float3 direction,
                                   simd::float3 color,
                                   float intensity)
    : m_direction(direction)
    , m_color(color)
    , m_intensity(intensity)
{
}

simd::float3 DirectionalLight::GetDirection() const
{
    return m_direction;
}

simd::float3 DirectionalLight::GetColor() const
{
    return m_color;
}

float DirectionalLight::GetIntensity() const
{
    return m_intensity;
}
