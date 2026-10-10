#pragma once

#include <simd/simd.h>
#include <cmath>

class MatrixFactory final
{
public:
    static inline simd::float4x4 Identity()
    {
        return matrix_identity_float4x4;
    }

    static inline simd::float4x4 Translation(simd::float3 t)
    {
        simd::float4x4 m = matrix_identity_float4x4;
        m.columns[3] = simd::float4{t.x, t.y, t.z, 1.0f};
        return m;
    }

    static inline simd::float4x4 UniformScale(float s)
    {
        simd::float4x4 m = matrix_identity_float4x4;
        m.columns[0].x = s;
        m.columns[1].y = s;
        m.columns[2].z = s;
        return m;
    }

    static inline simd::float4x4 ScaleXY(simd::float2 size)
    {
        simd::float4x4 m = matrix_identity_float4x4;
        m.columns[0].x = size.x;
        m.columns[1].y = size.y;
        return m;
    }

    static inline simd::float4x4 RotationX(float radians)
    {
        const float c = cosf(radians);
        const float s = sinf(radians);

        simd::float4x4 m = Identity();
        m.columns[1] = simd::float4{0.0f, c, s, 0.0f};
        m.columns[2] = simd::float4{0.0f, -s, c, 0.0f};
        return m;
    }

    static inline simd::float4x4 RotationY(float radians)
    {
        const float c = cosf(radians);
        const float s = sinf(radians);

        simd::float4x4 m = Identity();
        m.columns[0] = simd::float4{c, 0.0f, -s, 0.0f};
        m.columns[1] = simd::float4{0.0f, 1.0f, 0.0f, 0.0f};
        m.columns[2] = simd::float4{s, 0.0f, c, 0.0f};
        m.columns[3] = simd::float4{0.0f, 0.0f, 0.0f, 1.0f};
        return m;
    }

    static inline simd::float4x4 RotationZ(float radians)
    {
        const float c = cosf(radians);
        const float s = sinf(radians);

        simd::float4x4 m = Identity();
        m.columns[0] = simd::float4{c, s, 0.0f, 0.0f};
        m.columns[1] = simd::float4{-s, c, 0.0f, 0.0f};
        return m;
    }

    static inline simd::float4x4 BillboardFacingCamera(simd::float3 objectPos,
                                                       simd::float3 cameraPos)
    {
        simd::float3 forward = cameraPos - objectPos;
        if (simd::length_squared(forward) < 1e-6f)
        {
            forward = simd::float3{0.0f, 0.0f, 1.0f};
        }
        forward = simd::normalize(forward);

        simd::float3 upHint = simd::float3{0.0f, 1.0f, 0.0f};
        simd::float3 right = simd::cross(upHint, forward);
        if (simd::length_squared(right) < 1e-6f)
        {
            upHint = simd::float3{1.0f, 0.0f, 0.0f};
            right = simd::cross(upHint, forward);
        }
        right = simd::normalize(right);
        const simd::float3 up = simd::normalize(simd::cross(forward, right));

        simd::float4x4 m = matrix_identity_float4x4;
        m.columns[0] = simd::float4{right.x, right.y, right.z, 0.0f};
        m.columns[1] = simd::float4{up.x, up.y, up.z, 0.0f};
        m.columns[2] = simd::float4{forward.x, forward.y, forward.z, 0.0f};
        return m;
    }

    static inline simd::float4x4 FenceWorld(simd::float3 pos,
                                            simd::float2 size,
                                            float yaw,
                                            float pitch,
                                            float roll)
    {
        const simd::float4x4 rot =
            simd_mul(simd_mul(RotationY(yaw), RotationX(pitch)), RotationZ(roll));
        return simd_mul(simd_mul(Translation(pos), rot), ScaleXY(size));
    }

    static inline simd::float4x4 PerspectiveRH(float fovyRadians,
                                               float aspect,
                                               float zn,
                                               float zf)
    {
        const float ys = 1.0f / tanf(fovyRadians * 0.5f);
        const float xs = ys / aspect;
        const float zs = zf / (zn - zf);

        simd::float4x4 m{};
        m.columns[0] = simd::float4{xs, 0.0f, 0.0f, 0.0f};
        m.columns[1] = simd::float4{0.0f, ys, 0.0f, 0.0f};
        m.columns[2] = simd::float4{0.0f, 0.0f, zs, -1.0f};
        m.columns[3] = simd::float4{0.0f, 0.0f, zs * zn, 0.0f};
        return m;
    }

    static inline simd::float4x4 LookAtRH(simd::float3 eye,
                                          simd::float3 at,
                                          simd::float3 up)
    {
        const simd::float3 z = simd::normalize(eye - at);
        const simd::float3 x = simd::normalize(simd::cross(up, z));
        const simd::float3 y = simd::cross(z, x);

        simd::float4x4 m = matrix_identity_float4x4;
        m.columns[0] = simd::float4{x.x, y.x, z.x, 0.0f};
        m.columns[1] = simd::float4{x.y, y.y, z.y, 0.0f};
        m.columns[2] = simd::float4{x.z, y.z, z.z, 0.0f};
        m.columns[3] = simd::float4{
            -simd::dot(x, eye),
            -simd::dot(y, eye),
            -simd::dot(z, eye),
            1.0f
        };
        return m;
    }

    static inline simd::float4x4 OrthoRH(float left,
                                         float right,
                                         float bottom,
                                         float top,
                                         float zn,
                                         float zf)
    {
        simd::float4x4 m{};
        m.columns[0] = simd::float4{2.0f / (right - left), 0.0f, 0.0f, 0.0f};
        m.columns[1] = simd::float4{0.0f, 2.0f / (top - bottom), 0.0f, 0.0f};
        m.columns[2] = simd::float4{0.0f, 0.0f, 1.0f / (zn - zf), 0.0f};
        m.columns[3] = simd::float4{
            -(right + left) / (right - left),
            -(top + bottom) / (top - bottom),
            zn / (zn - zf),
            1.0f
        };
        return m;
    }
};
