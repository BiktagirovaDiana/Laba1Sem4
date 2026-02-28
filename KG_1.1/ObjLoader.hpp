#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct VertexPNT
{
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
};

struct ObjMaterial
{
    float kd[3] = {1.0f, 1.0f, 1.0f};
    float ks[3] = {0.0f, 0.0f, 0.0f};
    float ns = 32.0f;
    float d = 1.0f;
    std::string diffuseTexPath;
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

class ObjLoader
{
public:
    static bool LoadMesh(const std::string& path, ObjMesh& outMesh);
};
