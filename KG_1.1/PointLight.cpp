#include "PointLight.hpp"

PointLight::PointLight(simd::float3 position,
                       float range,
                       simd::float3 color,
                       float intensity)
    : m_position(position)
    , m_range(range)
    , m_color(color)
    , m_intensity(intensity)
{
}

simd::float3 PointLight::GetPosition() const
{
    return m_position;
}

float PointLight::GetRange() const
{
    return m_range;
}

simd::float3 PointLight::GetColor() const
{
    return m_color;
}

float PointLight::GetIntensity() const
{
    return m_intensity;
}
