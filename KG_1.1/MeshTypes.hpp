#pragma once

#include <simd/simd.h>
#include <cstdint>
#include <string>
#include <vector>

struct VertexPNT
{
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
};

struct ObjMaterial
{
    std::string name;
    float kd[3] = {1.0f, 1.0f, 1.0f};
    float ks[3] = {0.0f, 0.0f, 0.0f};
    float ns = 32.0f;
    float d = 1.0f;
    std::string diffuseTexPath;
    std::string normalTexPath;
    std::string heightTexPath;
};

struct ObjSubmesh
{
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    uint32_t materialIndex = 0;
};

struct ObjMesh
{
    std::vector<VertexPNT> vertices;
    std::vector<uint32_t> indices;
    std::vector<ObjMaterial> materials;
    std::vector<ObjSubmesh> submeshes;
};

struct MaterialGPU
{
    simd::float4 kd_ns = {1.0f, 1.0f, 1.0f, 32.0f};
    simd::float4 ks_alpha = {0.0f, 0.0f, 0.0f, 1.0f};
    simd::float2 uvScale = {1.0f, 1.0f};
    simd::float2 uvSpeed = {0.0f, 0.0f};
    simd::uint4 textureFlags = {0u, 0u, 0u, 0u};
    simd::float4 detailParams = {2.0f, 1.0f, 0.0f, 0.0f};
};
