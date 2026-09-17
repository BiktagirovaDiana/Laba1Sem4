#import "SceneAssetLoader.hpp"
#import <Foundation/Foundation.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <string>
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
}

static bool FileExists(const char* p)
{
    struct stat st;
    return (stat(p, &st) == 0) && S_ISREG(st.st_mode);
}

static bool FileExists(const std::string& path)
{
    FILE* file = fopen(path.c_str(), "rb");
    if (!file)
    {
        return false;
    }

    fclose(file);
    return true;
}

static bool DirExists(const char* p)
{
    struct stat st;
    return (stat(p, &st) == 0) && S_ISDIR(st.st_mode);
}

static std::string DirName(const std::string& p)
{
    const size_t slashPos = p.find_last_of("/\\");
    if (slashPos == std::string::npos)
    {
        return std::string();
    }
    return p.substr(0, slashPos);
}

static std::string GetBaseDir(const std::string& path)
{
    const size_t slashPos = path.find_last_of("/\\");
    if (slashPos == std::string::npos)
    {
        return std::string();
    }
    return path.substr(0, slashPos + 1);
}

static std::string GetFileName(const std::string& path)
{
    const size_t slashPos = path.find_last_of("/\\");
    if (slashPos == std::string::npos)
    {
        return path;
    }
    return path.substr(slashPos + 1);
}

static bool IsAbsolutePath(const std::string& path)
{
    if (path.empty())
    {
        return false;
    }

    if (path[0] == '/' || path[0] == '\\')
    {
        return true;
    }

    return path.size() > 1 && path[1] == ':';
}

static std::string ToLowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return (char)std::tolower(ch); });
    return value;
}

static std::string ResolveSiblingTexture(const std::string& baseDir, const std::string& fileName)
{
    if (baseDir.empty() || fileName.empty())
    {
        return std::string();
    }

    const std::string fullPath = baseDir + fileName;
    return FileExists(fullPath) ? fullPath : std::string();
}

static bool LooksLikeHeightTexture(const std::string& texName)
{
    if (texName.empty())
    {
        return false;
    }

    const std::string lower = ToLowerCopy(texName);
    return lower.find("height") != std::string::npos ||
           lower.find("disp") != std::string::npos ||
           lower.find("displacement") != std::string::npos;
}

static std::string ResolveTexturePath(const std::string& baseDir, const std::string& texName)
{
    if (texName.empty())
    {
        return std::string();
    }

    std::string normalized = texName;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');

    if (IsAbsolutePath(normalized) || baseDir.empty())
    {
        return normalized;
    }

    const std::string directPath = baseDir + normalized;
    if (FileExists(directPath))
    {
        return directPath;
    }

    const size_t slashPos = normalized.find_last_of('/');
    const std::string fileNameOnly =
        (slashPos == std::string::npos) ? normalized : normalized.substr(slashPos + 1);
    const std::string siblingPath = baseDir + fileNameOnly;
    if (FileExists(siblingPath))
    {
        return siblingPath;
    }

    const size_t dotPos = fileNameOnly.find_last_of('.');
    if (dotPos != std::string::npos)
    {
        const std::string stem = fileNameOnly.substr(0, dotPos);
        const char* extensions[] = {".jpg", ".jpeg", ".png", ".tga"};
        for (const char* ext : extensions)
        {
            const std::string candidate = baseDir + stem + ext;
            if (FileExists(candidate))
            {
                return candidate;
            }
        }
    }

    return siblingPath;
}

static std::string JoinPath(const std::string& a, const std::string& b)
{
    if (a.empty())
    {
        return b;
    }
    if (a.back() == '/')
    {
        return a + b;
    }
    return a + "/" + b;
}

static std::vector<std::string> GetAssetCandidateDirs()
{
    std::vector<std::string> candidateDirs;

    const std::string sourceDir = DirName(__FILE__);
    if (!sourceDir.empty())
    {
        candidateDirs.push_back(JoinPath(sourceDir, "assets"));
    }

    candidateDirs.push_back("assets");
    candidateDirs.push_back("KG_1.1/assets");

    NSString* exePathNs = [[NSBundle mainBundle] executablePath];
    if (exePathNs)
    {
        const std::string exeDir = DirName([exePathNs UTF8String]);
        if (!exeDir.empty())
        {
            candidateDirs.push_back(JoinPath(exeDir, "assets"));
            candidateDirs.push_back(JoinPath(exeDir, "../assets"));
            candidateDirs.push_back(JoinPath(exeDir, "../Resources/assets"));
        }
    }

    return candidateDirs;
}

static bool ParseModelAssetName(const std::string& fileName, int& sortIndex)
{
    if (fileName == "woodroot.obj" || fileName == "wootroot.obj")
    {
        sortIndex = 5;
        return true;
    }
    if (fileName == "cerberusobj.obj")
    {
        sortIndex = 6;
        return true;
    }

    constexpr const char* kPrefix = "model";
    constexpr const char* kSuffix = ".obj";
    if (fileName.size() < 9 || fileName.rfind(kSuffix) != fileName.size() - 4)
    {
        return false;
    }
    if (fileName.compare(0, 5, kPrefix) != 0)
    {
        return false;
    }

    const std::string middle = fileName.substr(5, fileName.size() - 9);
    if (middle.empty())
    {
        sortIndex = 1;
        return true;
    }

    for (char ch : middle)
    {
        if (!std::isdigit(static_cast<unsigned char>(ch)))
        {
            return false;
        }
    }

    sortIndex = std::max(2, std::stoi(middle));
    return true;
}

static std::vector<std::string> ResolveModelAssetPaths()
{
    std::vector<std::string> modelPaths;
    NSFileManager* fileManager = [NSFileManager defaultManager];

    for (const std::string& dir : GetAssetCandidateDirs())
    {
        if (!DirExists(dir.c_str()))
        {
            continue;
        }

        NSString* nsDir = [NSString stringWithUTF8String:dir.c_str()];
        NSError* err = nil;
        NSArray<NSString*>* contents = [fileManager contentsOfDirectoryAtPath:nsDir error:&err];
        if (!contents)
        {
            NSLog(@"Failed to enumerate assets in %@: %@", nsDir, err);
            continue;
        }

        std::vector<std::pair<int, std::string>> foundModels;
        foundModels.reserve(contents.count);
        for (NSString* entry in contents)
        {
            const std::string fileName = [entry UTF8String];
            int sortIndex = 0;
            if (!ParseModelAssetName(fileName, sortIndex))
            {
                continue;
            }

            const std::string fullPath = JoinPath(dir, fileName);
            if (!FileExists(fullPath.c_str()))
            {
                continue;
            }

            foundModels.push_back({sortIndex, fullPath});
        }

        if (foundModels.empty())
        {
            continue;
        }

        std::sort(foundModels.begin(), foundModels.end(),
                  [](const auto& lhs, const auto& rhs)
                  {
                      if (lhs.first != rhs.first)
                      {
                          return lhs.first < rhs.first;
                      }
                      return lhs.second < rhs.second;
                  });

        modelPaths.reserve(foundModels.size());
        for (const auto& entry : foundModels)
        {
            modelPaths.push_back(entry.second);
        }
        return modelPaths;
    }

    return modelPaths;
}

static bool LoadObjMesh(const std::string& path, ObjMesh& outMesh)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> tinyMaterials;
    std::string warn, err;

    const std::string baseDir = GetBaseDir(path);
    const bool ok = tinyobj::LoadObj(
        &attrib, &shapes, &tinyMaterials, &warn, &err, path.c_str(), baseDir.c_str(), true);
    if (!warn.empty())
    {
        fprintf(stderr, "OBJ warning: %s\n", warn.c_str());
    }
    if (!err.empty())
    {
        fprintf(stderr, "OBJ error: %s\n", err.c_str());
    }
    if (!ok)
    {
        return false;
    }

    outMesh = ObjMesh{};
    outMesh.materials.reserve(tinyMaterials.size() + 1);

    for (const auto& tm : tinyMaterials)
    {
        ObjMaterial m;
        m.name = tm.name;
        m.kd[0] = tm.diffuse[0];
        m.kd[1] = tm.diffuse[1];
        m.kd[2] = tm.diffuse[2];
        m.ks[0] = tm.specular[0];
        m.ks[1] = tm.specular[1];
        m.ks[2] = tm.specular[2];
        m.ns = tm.shininess;
        m.d = tm.dissolve;

        const std::string primaryColorTex =
            !tm.diffuse_texname.empty() ? tm.diffuse_texname :
            !tm.ambient_texname.empty() ? tm.ambient_texname :
            tm.specular_texname;
        m.diffuseTexPath = ResolveTexturePath(baseDir, primaryColorTex);
        m.normalTexPath = ResolveTexturePath(baseDir, tm.normal_texname);

        std::string displacementTexName = tm.displacement_texname;
        if (displacementTexName.empty() && LooksLikeHeightTexture(tm.bump_texname))
        {
            displacementTexName = tm.bump_texname;
        }
        m.heightTexPath = ResolveTexturePath(baseDir, displacementTexName);

        outMesh.materials.push_back(m);
    }

    const std::string modelFileName = ToLowerCopy(GetFileName(path));
    for (ObjMaterial& material : outMesh.materials)
    {
        if (modelFileName == "cerberusobj.obj")
        {
            if (material.diffuseTexPath.empty() || !FileExists(material.diffuseTexPath))
            {
                material.diffuseTexPath = ResolveSiblingTexture(baseDir, "Cerberus_A.jpg");
            }
            if (material.normalTexPath.empty())
            {
                material.normalTexPath = ResolveSiblingTexture(baseDir, "Cerberus_N.jpg");
            }
        }
    }

    ObjMaterial defaultMaterial{};
    if (modelFileName == "woodroot.obj" || modelFileName == "wootroot.obj")
    {
        defaultMaterial.name = "woodroot";
        defaultMaterial.diffuseTexPath = ResolveSiblingTexture(baseDir, "Aset_wood_root_M_rkswd_2K_Albedo.jpg");
        defaultMaterial.normalTexPath = ResolveSiblingTexture(baseDir, "Aset_wood_root_M_rkswd_2K_Normal_LOD0.jpg");
    }

    const uint32_t defaultMaterialIndex = (uint32_t)outMesh.materials.size();
    outMesh.materials.push_back(defaultMaterial);

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

bool SceneAssetLoader::Load(const SceneLoadConfig& config,
                            TextureLoader textureLoader,
                            SceneLoadResult& outScene) const
{
    char cwd[2048];
    getcwd(cwd, sizeof(cwd));
    NSLog(@"CWD = %s", cwd);

    const std::vector<std::string> objPaths = ResolveModelAssetPaths();
    if (objPaths.empty())
    {
        NSLog(@"No model OBJ files found. Expected names like model.obj, model2.obj, model3.obj in assets.");
        outScene = SceneLoadResult{};
        return false;
    }

    outScene = SceneLoadResult{};
    bool haveBounds = false;

    for (size_t modelIndex = 0; modelIndex < objPaths.size(); ++modelIndex)
    {
        const std::string& objPath = objPaths[modelIndex];
        ObjMesh mesh;
        NSLog(@"Loading OBJ from: %s", objPath.c_str());
        const bool ok = LoadObjMesh(objPath, mesh);

        if (!ok || mesh.vertices.empty() || mesh.indices.empty())
        {
            NSLog(@"OBJ load failed OR empty mesh. path=%s vertices=%lu indices=%lu",
                  objPath.c_str(),
                  (unsigned long)mesh.vertices.size(),
                  (unsigned long)mesh.indices.size());
            continue;
        }

        const uint32_t vertexBase = (uint32_t)outScene.vertices.size();
        const uint32_t indexBase = (uint32_t)outScene.indices.size();
        const uint32_t materialBase = (uint32_t)outScene.materials.size();
        simd::float3 modelAabbMin = simd::float3{mesh.vertices[0].px, mesh.vertices[0].py, mesh.vertices[0].pz};
        simd::float3 modelAabbMax = modelAabbMin;

        if (!haveBounds)
        {
            outScene.meshAabbMin = modelAabbMin;
            outScene.meshAabbMax = modelAabbMin;
            haveBounds = true;
        }

        for (const VertexPNT& v : mesh.vertices)
        {
            outScene.vertices.push_back(v);
            if (v.px < modelAabbMin.x) modelAabbMin.x = v.px;
            if (v.py < modelAabbMin.y) modelAabbMin.y = v.py;
            if (v.pz < modelAabbMin.z) modelAabbMin.z = v.pz;
            if (v.px > modelAabbMax.x) modelAabbMax.x = v.px;
            if (v.py > modelAabbMax.y) modelAabbMax.y = v.py;
            if (v.pz > modelAabbMax.z) modelAabbMax.z = v.pz;
            if (v.px < outScene.meshAabbMin.x) outScene.meshAabbMin.x = v.px;
            if (v.py < outScene.meshAabbMin.y) outScene.meshAabbMin.y = v.py;
            if (v.pz < outScene.meshAabbMin.z) outScene.meshAabbMin.z = v.pz;
            if (v.px > outScene.meshAabbMax.x) outScene.meshAabbMax.x = v.px;
            if (v.py > outScene.meshAabbMax.y) outScene.meshAabbMax.y = v.py;
            if (v.pz > outScene.meshAabbMax.z) outScene.meshAabbMax.z = v.pz;
        }

        outScene.indices.reserve(outScene.indices.size() + mesh.indices.size());
        for (uint32_t idx : mesh.indices)
        {
            outScene.indices.push_back(vertexBase + idx);
        }

        outScene.materials.reserve(outScene.materials.size() + mesh.materials.size());
        outScene.diffuseTextures.reserve(outScene.diffuseTextures.size() + mesh.materials.size());
        outScene.normalTextures.reserve(outScene.normalTextures.size() + mesh.materials.size());
        outScene.heightTextures.reserve(outScene.heightTextures.size() + mesh.materials.size());
        for (const ObjMaterial& m : mesh.materials)
        {
            MaterialGPU gpuMat;
            gpuMat.kd_ns = simd::float4{m.kd[0], m.kd[1], m.kd[2], (m.ns > 0.0f) ? m.ns : 32.0f};
            gpuMat.ks_alpha = simd::float4{m.ks[0], m.ks[1], m.ks[2], m.d};
            gpuMat.uvScale = config.textureTiling;
            gpuMat.uvSpeed = config.textureScrollSpeed;
            gpuMat.detailParams = simd::float4{outScene.meshRadius * config.tessellationStrength, 1.0f, 0.0f, 0.0f};

            id<MTLTexture> diffuseTex = nil;
            if (!m.diffuseTexPath.empty() && textureLoader)
            {
                diffuseTex = textureLoader(m.diffuseTexPath, true);
            }
            id<MTLTexture> normalTex = nil;
            if (!m.normalTexPath.empty() && textureLoader)
            {
                normalTex = textureLoader(m.normalTexPath, false);
            }
            id<MTLTexture> heightTex = nil;
            if (!m.heightTexPath.empty() && textureLoader)
            {
                heightTex = textureLoader(m.heightTexPath, false);
            }

            gpuMat.textureFlags[0] = diffuseTex ? 1u : 0u;
            gpuMat.textureFlags[1] = normalTex ? 1u : 0u;
            gpuMat.textureFlags[2] = heightTex ? 1u : 0u;
            gpuMat.textureFlags[3] = 0u;

            NSLog(@"Material '%s': diffuseTex=%s normalTex=%s heightTex=%s",
                  m.name.empty() ? "<default>" : m.name.c_str(),
                  m.diffuseTexPath.empty() ? "<none>" : m.diffuseTexPath.c_str(),
                  m.normalTexPath.empty() ? "<none>" : m.normalTexPath.c_str(),
                  m.heightTexPath.empty() ? "<none>" : m.heightTexPath.c_str());

            outScene.materials.push_back(gpuMat);
            outScene.diffuseTextures.push_back(diffuseTex);
            outScene.normalTextures.push_back(normalTex);
            outScene.heightTextures.push_back(heightTex);
        }

        const uint32_t sourceModelIndex = (uint32_t)modelIndex;
        if (outScene.modelBounds.size() <= sourceModelIndex)
        {
            outScene.modelBounds.resize(sourceModelIndex + 1u);
        }
        SceneModelBounds modelBounds;
        modelBounds.localAabbMin = modelAabbMin;
        modelBounds.localAabbMax = modelAabbMax;
        modelBounds.localCenter = (modelAabbMin + modelAabbMax) * 0.5f;
        modelBounds.localRadius = simd::length((modelAabbMax - modelAabbMin) * 0.5f);
        if (modelBounds.localRadius < 1.0f)
        {
            modelBounds.localRadius = 1.0f;
        }
        outScene.modelBounds[sourceModelIndex] = modelBounds;

        const simd::float3 baseOffset = GetOffsetForModel(config, sourceModelIndex);
        const bool isModel4 = (sourceModelIndex == 3u);
        const int instanceCount = isModel4 ? config.model4InstanceCount : 1;
        const int gridSize = 100;
        const float spacing = 12.0f;

        outScene.batches.reserve(outScene.batches.size() + mesh.submeshes.size() * (size_t)instanceCount);
        for (int instanceIndex = 0; instanceIndex < instanceCount; ++instanceIndex)
        {
            simd::float3 instanceOffset = baseOffset;
            if (isModel4)
            {
                const int row = instanceIndex / gridSize;
                const int col = instanceIndex % gridSize;
                instanceOffset.x += (float)col * spacing;
                instanceOffset.z += (float)row * spacing;
            }

            SceneInstance instance;
            instance.sourceModelIndex = sourceModelIndex;
            instance.worldOffset = instanceOffset;
            instance.scale = GetScaleForModel(config, sourceModelIndex);
            instance.worldAabbMin = instanceOffset + modelBounds.localAabbMin * instance.scale;
            instance.worldAabbMax = instanceOffset + modelBounds.localAabbMax * instance.scale;
            instance.worldCenter = instanceOffset + modelBounds.localCenter * instance.scale;
            instance.worldRadius = modelBounds.localRadius * instance.scale;
            const uint32_t sceneInstanceIndex = (uint32_t)outScene.instances.size();
            outScene.instances.push_back(instance);
            outScene.model4PlaneStates.push_back(0u);

            SceneCollisionAabb collisionAabb;
            collisionAabb.min = instance.worldAabbMin;
            collisionAabb.max = instance.worldAabbMax;
            outScene.collisionAabbs.push_back(collisionAabb);

            for (const ObjSubmesh& sm : mesh.submeshes)
            {
                SceneDrawBatch b;
                b.indexOffset = indexBase + sm.indexOffset;
                b.indexCount = sm.indexCount;
                b.materialIndex = materialBase + sm.materialIndex;
                b.sourceModelIndex = sourceModelIndex;
                b.instanceIndex = sceneInstanceIndex;
                outScene.batches.push_back(b);
            }
        }
    }

    outScene.indexCount = (uint32_t)outScene.indices.size();
    if (outScene.indexCount == 0 || outScene.vertices.empty())
    {
        NSLog(@"No valid OBJ meshes were loaded.");
        return false;
    }

    outScene.meshCenter = (outScene.meshAabbMin + outScene.meshAabbMax) * 0.5f;
    const simd::float3 extents = (outScene.meshAabbMax - outScene.meshAabbMin) * 0.5f;
    outScene.meshRadius = simd::length(extents);
    if (outScene.meshRadius < 1.0f)
    {
        outScene.meshRadius = 1.0f;
    }

    return true;
}

float SceneAssetLoader::GetScaleForModel(const SceneLoadConfig& config, uint32_t modelIndex) const
{
    if (modelIndex < config.modelScales.size())
    {
        return config.modelScales[modelIndex];
    }
    return 1.0f;
}

simd::float3 SceneAssetLoader::GetOffsetForModel(const SceneLoadConfig& config, uint32_t modelIndex) const
{
    if (modelIndex < config.modelOffsets.size())
    {
        return config.modelOffsets[modelIndex];
    }
    return simd::float3{0.0f, 0.0f, 0.0f};
}
