#include "ObjLoader.hpp"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <unordered_map>
#include <vector>

namespace
{
struct Key
{
    int v;
    int n;
    int t;
    bool operator==(const Key& o) const
    {
        return v == o.v && n == o.n && t == o.t;
    }
};

struct KeyHash
{
    size_t operator()(const Key& k) const
    {
        return (size_t)k.v * 73856093u ^ (size_t)k.n * 19349663u ^ (size_t)k.t * 83492791u;
    }
};

std::string GetBaseDir(const std::string& path)
{
    const size_t slashPos = path.find_last_of("/\\");
    if (slashPos == std::string::npos)
    {
        return std::string();
    }
    return path.substr(0, slashPos + 1);
}
}

bool ObjLoader::LoadMesh(const std::string& path, ObjMesh& outMesh)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> tinyMaterials;
    std::string warn, err;

    const std::string baseDir = GetBaseDir(path);
    const bool ok = tinyobj::LoadObj(
        &attrib, &shapes, &tinyMaterials, &warn, &err, path.c_str(), baseDir.c_str(), true);
    if (!ok)
    {
        return false;
    }

    outMesh = ObjMesh{};
    outMesh.materials.reserve(tinyMaterials.size() + 1);

    for (const auto& tm : tinyMaterials)
    {
        ObjMaterial m;
        m.kd[0] = tm.diffuse[0];
        m.kd[1] = tm.diffuse[1];
        m.kd[2] = tm.diffuse[2];
        m.ks[0] = tm.specular[0];
        m.ks[1] = tm.specular[1];
        m.ks[2] = tm.specular[2];
        m.ns = tm.shininess;
        m.d = tm.dissolve;
        if (!tm.diffuse_texname.empty())
        {
            m.diffuseTexPath = baseDir + tm.diffuse_texname;
        }
        outMesh.materials.push_back(m);
    }

    const uint32_t defaultMaterialIndex = (uint32_t)outMesh.materials.size();
    outMesh.materials.push_back(ObjMaterial{});

    std::unordered_map<Key, uint32_t, KeyHash> uniqueVerts;
    std::vector<std::vector<uint32_t>> perMaterialIndices(outMesh.materials.size());

    for (const auto& s : shapes)
    {
        size_t indexOffset = 0;
        for (size_t f = 0; f < s.mesh.num_face_vertices.size(); ++f)
        {
            const int fv = (int)s.mesh.num_face_vertices[f];
            int matId = -1;
            if (f < s.mesh.material_ids.size())
            {
                matId = s.mesh.material_ids[f];
            }
            uint32_t matIndex = defaultMaterialIndex;
            if (matId >= 0 && (size_t)matId < tinyMaterials.size())
            {
                matIndex = (uint32_t)matId;
            }

            for (int v = 0; v < fv; ++v)
            {
                const tinyobj::index_t idx = s.mesh.indices[indexOffset + (size_t)v];
                const Key key{idx.vertex_index, idx.normal_index, idx.texcoord_index};

                auto it = uniqueVerts.find(key);
                if (it != uniqueVerts.end())
                {
                    perMaterialIndices[matIndex].push_back(it->second);
                    continue;
                }

                VertexPNT vert{};
                const int vi = idx.vertex_index * 3;
                vert.px = attrib.vertices[(size_t)vi + 0];
                vert.py = attrib.vertices[(size_t)vi + 1];
                vert.pz = attrib.vertices[(size_t)vi + 2];

                if (idx.normal_index >= 0)
                {
                    const int ni = idx.normal_index * 3;
                    vert.nx = attrib.normals[(size_t)ni + 0];
                    vert.ny = attrib.normals[(size_t)ni + 1];
                    vert.nz = attrib.normals[(size_t)ni + 2];
                }
                else
                {
                    vert.nx = 0.0f;
                    vert.ny = 1.0f;
                    vert.nz = 0.0f;
                }

                if (idx.texcoord_index >= 0)
                {
                    const int ti = idx.texcoord_index * 2;
                    vert.u = attrib.texcoords[(size_t)ti + 0];
                    vert.v = attrib.texcoords[(size_t)ti + 1];
                }
                else
                {
                    vert.u = 0.0f;
                    vert.v = 0.0f;
                }

                const uint32_t newIndex = (uint32_t)outMesh.vertices.size();
                outMesh.vertices.push_back(vert);
                uniqueVerts[key] = newIndex;
                perMaterialIndices[matIndex].push_back(newIndex);
            }

            indexOffset += (size_t)fv;
        }
    }

    uint32_t runningOffset = 0;
    for (uint32_t i = 0; i < (uint32_t)perMaterialIndices.size(); ++i)
    {
        const auto& matIndices = perMaterialIndices[i];
        if (matIndices.empty())
        {
            continue;
        }

        ObjSubmesh sm;
        sm.indexOffset = runningOffset;
        sm.indexCount = (uint32_t)matIndices.size();
        sm.materialIndex = i;
        outMesh.submeshes.push_back(sm);

        outMesh.indices.insert(outMesh.indices.end(), matIndices.begin(), matIndices.end());
        runningOffset += sm.indexCount;
    }

    return !outMesh.vertices.empty() && !outMesh.indices.empty();
}
