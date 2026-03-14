#include "SpotLight.hpp"

SpotLight::SpotLight(simd::float3 position,
                     simd::float3 direction,
                     float range,
                     float coneAngleDegrees,
                     simd::float3 color,
                     float intensity)
    : m_position(position)
    , m_direction(direction)
    , m_range(range)
    , m_coneAngleRadians(coneAngleDegrees * (float)M_PI / 180.0f)
    , m_color(color)
    , m_intensity(intensity)
{
}

simd::float3 SpotLight::GetPosition() const
{
    return m_position;
}

simd::float3 SpotLight::GetDirection() const
{
    return m_direction;
}

float SpotLight::GetRange() const
{
    return m_range;
}

float SpotLight::GetConeAngleRadians() const
{
    return m_coneAngleRadians;
}

simd::float3 SpotLight::GetColor() const
{
    return m_color;
}

float SpotLight::GetIntensity() const
{
    return m_intensity;
}
