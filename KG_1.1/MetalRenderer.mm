//
//  MetalRenderer.mm

//  KG_1.1
//
//  Created by Macbook on 21.02.2026.
//

#import "MetalRenderer.hpp"
#include <simd/simd.h>
#include "ObjLoader.hpp"
#include "InputDevice.hpp"
#import <Foundation/Foundation.h>

#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

static inline simd_float4x4 MatIdentity()
{
    return matrix_identity_float4x4;
}

static inline simd_float4x4 MatRotationY(float a)
{
    const float c = cosf(a);
    const float s = sinf(a);

    simd_float4x4 m = MatIdentity();
    m.columns[0] = (simd_float4){  c, 0.0f, -s, 0.0f };
    m.columns[1] = (simd_float4){ 0.0f, 1.0f, 0.0f, 0.0f };
    m.columns[2] = (simd_float4){  s, 0.0f,  c, 0.0f };
    m.columns[3] = (simd_float4){ 0.0f, 0.0f, 0.0f, 1.0f };
    return m;
}

static inline simd_float4x4 MatLookAtRH(simd_float3 eye, simd_float3 at, simd_float3 up)
{
    simd_float3 z = simd_normalize(eye - at);
    simd_float3 x = simd_normalize(simd_cross(up, z));
    simd_float3 y = simd_cross(z, x);

    simd_float4x4 m;
    m.columns[0] = (simd_float4){ x.x, y.x, z.x, 0.0f };
    m.columns[1] = (simd_float4){ x.y, y.y, z.y, 0.0f };
    m.columns[2] = (simd_float4){ x.z, y.z, z.z, 0.0f };
    m.columns[3] = (simd_float4){
        -simd_dot(x, eye),
        -simd_dot(y, eye),
        -simd_dot(z, eye),
         1.0f
    };
    return m;
}

static inline simd_float4x4 MatPerspectiveRH(float fovyRadians, float aspect, float zn, float zf)
{
    const float ys = 1.0f / tanf(fovyRadians * 0.5f);
    const float xs = ys / aspect;
    const float zs = zf / (zn - zf);

    simd_float4x4 m = {};
    m.columns[0] = (simd_float4){ xs, 0.0f, 0.0f, 0.0f };
    m.columns[1] = (simd_float4){ 0.0f, ys, 0.0f, 0.0f };
    m.columns[2] = (simd_float4){ 0.0f, 0.0f, zs, -1.0f };
    m.columns[3] = (simd_float4){ 0.0f, 0.0f, zs * zn, 0.0f };
    return m;
}

static inline simd_float4x4 MatTranslation(simd_float3 t)
{
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[3] = (simd_float4){ t.x, t.y, t.z, 1.0f };
    return m;
}

static inline simd_float4x4 MatScale(float s)
{
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[0].x = s;
    m.columns[1].y = s;
    m.columns[2].z = s;
    return m;
}

static inline simd_float4x4 MatBillboardFacingCamera(simd_float3 objectPos, simd_float3 cameraPos)
{
    simd_float3 forward = cameraPos - objectPos;
    if (simd_length_squared(forward) < 1e-6f)
    {
        forward = simd_float3{0.0f, 0.0f, 1.0f};
    }
    forward = simd_normalize(forward);

    simd_float3 upHint = simd_float3{0.0f, 1.0f, 0.0f};
    simd_float3 right = simd_cross(upHint, forward);
    if (simd_length_squared(right) < 1e-6f)
    {
        upHint = simd_float3{1.0f, 0.0f, 0.0f};
        right = simd_cross(upHint, forward);
    }
    right = simd_normalize(right);
    const simd_float3 up = simd_normalize(simd_cross(forward, right));

    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[0] = simd_float4{right.x, right.y, right.z, 0.0f};
    m.columns[1] = simd_float4{up.x, up.y, up.z, 0.0f};
    m.columns[2] = simd_float4{forward.x, forward.y, forward.z, 0.0f};
    return m;
}

static bool IsSphereVisibleInFrustum(const simd::float4x4& viewMatrix,
                                     simd::float3 worldCenter,
                                     float radius,
                                     float nearPlane,
                                     float farPlane,
                                     float tanHalfFovX,
                                     float tanHalfFovY)
{
    const simd::float4 centerView4 = viewMatrix * simd::float4{worldCenter.x, worldCenter.y, worldCenter.z, 1.0f};
    const simd::float3 centerView = simd::float3{centerView4.x, centerView4.y, centerView4.z};
    const float depth = -centerView.z;

    if (depth + radius < nearPlane)
    {
        return false;
    }
    if (depth - radius > farPlane)
    {
        return false;
    }
    if (fabsf(centerView.x) > depth * tanHalfFovX + radius)
    {
        return false;
    }
    if (fabsf(centerView.y) > depth * tanHalfFovY + radius)
    {
        return false;
    }

    return true;
}

static simd::float3 MinVec3(simd::float3 a, simd::float3 b)
{
    return simd::float3{fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
}

static simd::float3 MaxVec3(simd::float3 a, simd::float3 b)
{
    return simd::float3{fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
}

static float ComponentForAxis(simd::float3 v, int axis)
{
    if (axis == 0) return v.x;
    if (axis == 1) return v.y;
    return v.z;
}

static bool FileExists(const char* p)
{
    struct stat st;
    return (stat(p, &st) == 0) && S_ISREG(st.st_mode);
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

static simd::float3 Min3(simd::float3 a, simd::float3 b, simd::float3 c)
{
    return simd::float3{fminf(a.x, fminf(b.x, c.x)),
                        fminf(a.y, fminf(b.y, c.y)),
                        fminf(a.z, fminf(b.z, c.z))};
}

static simd::float3 Max3(simd::float3 a, simd::float3 b, simd::float3 c)
{
    return simd::float3{fmaxf(a.x, fmaxf(b.x, c.x)),
                        fmaxf(a.y, fmaxf(b.y, c.y)),
                        fmaxf(a.z, fmaxf(b.z, c.z))};
}

static std::string ResolveAssetPath(const std::string& fileName)
{
    for (const std::string& d : GetAssetCandidateDirs())
    {
        const std::string full = JoinPath(d, fileName);
        if (FileExists(full.c_str()))
        {
            return full;
        }
    }

    return std::string();
}

static ObjMesh CreateTexturedPlaneMesh(const std::string& diffuseTexturePath)
{
    ObjMesh mesh;
    mesh.vertices =
    {
        VertexPNT{-0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        VertexPNT{ 0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        VertexPNT{ 0.5f,  0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f},
        VertexPNT{-0.5f,  0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}
    };
    mesh.indices = {0u, 1u, 2u, 0u, 2u, 3u};

    ObjMaterial material;
    material.name = "model4_plane";
    material.kd[0] = 1.0f;
    material.kd[1] = 1.0f;
    material.kd[2] = 1.0f;
    material.d = 1.0f;
    material.diffuseTexPath = diffuseTexturePath;
    mesh.materials.push_back(material);

    ObjSubmesh submesh;
    submesh.indexOffset = 0;
    submesh.indexCount = (uint32_t)mesh.indices.size();
    submesh.materialIndex = 0;
    mesh.submeshes.push_back(submesh);
    return mesh;
}


struct CameraCB
{
    simd::float4x4 world;
    simd::float4x4 view;
    simd::float4x4 proj;
    simd::float3   lightDir;
    float          lightIntensity;
    simd::float3   lightColor;
    float          pad0;
    simd::float3   cameraPos;
    float          timeSeconds;
};

static simd::float4x4 Identity()
{
    return matrix_identity_float4x4;
}

MetalRenderer::MetalRenderer(MTKView* view) : m_view(view)
{
    CreateDeviceAndSwapchain();
    m_gbuffer.SetDevice(m_device);
    CreateDepth();
    CreateShadersAndPSO();
    CreateConstantBuffer();
    CreateSamplerAndFallbackTexture();
    CreateModel4PlaneResources();
    m_camPos = simd::float3{0.0f, 0.0f, 3.0f};
    m_camSpeed = 120.0f;
    m_yaw = (float)M_PI;
    m_pitch = 0.0f;
    LoadObjMesh();
    m_directionalLight = DirectionalLight(simd::float3{-0.3f, -1.0f, -0.2f},
                                          simd::float3{1.0f, 1.0f, 1.0f},
                                          1.5f);
}
 
MetalRenderer::~MetalRenderer() {}

float MetalRenderer::GetTessellationStrengthForModel(uint32_t modelIndex) const
{
    if (modelIndex < m_modelTessellationStrengths.size())
    {
        return m_modelTessellationStrengths[modelIndex];
    }

    return m_tessellationStrength;
}

simd::float3 MetalRenderer::GetOffsetForModel(uint32_t modelIndex) const
{
    if (modelIndex < m_modelOffsets.size())
    {
        return m_modelOffsets[modelIndex];
    }

    return simd::float3{0.0f, 0.0f, 0.0f};
}


void MetalRenderer::BuildSceneBVH()
{
    m_bvhNodes.clear();
    m_bvhInstanceIndices.clear();
    m_visibleInstances.assign(m_sceneInstances.size(), 0u);
    if (m_sceneInstances.empty())
    {
        return;
    }

    m_bvhInstanceIndices.resize(m_sceneInstances.size());
    for (uint32_t i = 0; i < m_sceneInstances.size(); ++i)
    {
        m_bvhInstanceIndices[i] = i;
    }

    BuildSceneBVHNode(0u, (uint32_t)m_bvhInstanceIndices.size());
}

uint32_t MetalRenderer::BuildSceneBVHNode(uint32_t begin, uint32_t end)
{
    BvhNode node;
    node.firstInstance = begin;
    node.instanceCount = end - begin;

    const SceneInstance& firstInstance = m_sceneInstances[m_bvhInstanceIndices[begin]];
    simd::float3 boundsMin = firstInstance.worldAabbMin;
    simd::float3 boundsMax = firstInstance.worldAabbMax;
    simd::float3 centroidMin = firstInstance.worldCenter;
    simd::float3 centroidMax = firstInstance.worldCenter;

    for (uint32_t i = begin + 1; i < end; ++i)
    {
        const SceneInstance& instance = m_sceneInstances[m_bvhInstanceIndices[i]];
        boundsMin = MinVec3(boundsMin, instance.worldAabbMin);
        boundsMax = MaxVec3(boundsMax, instance.worldAabbMax);
        centroidMin = MinVec3(centroidMin, instance.worldCenter);
        centroidMax = MaxVec3(centroidMax, instance.worldCenter);
    }

    node.aabbMin = boundsMin;
    node.aabbMax = boundsMax;

    const uint32_t nodeIndex = (uint32_t)m_bvhNodes.size();
    m_bvhNodes.push_back(node);

    const uint32_t instanceCount = end - begin;
    if (instanceCount <= 16u)
    {
        m_bvhNodes[nodeIndex].isLeaf = true;
        return nodeIndex;
    }

    const simd::float3 centroidExtent = centroidMax - centroidMin;
    int splitAxis = 0;
    if (centroidExtent.y > centroidExtent.x && centroidExtent.y >= centroidExtent.z)
    {
        splitAxis = 1;
    }
    else if (centroidExtent.z > centroidExtent.x && centroidExtent.z >= centroidExtent.y)
    {
        splitAxis = 2;
    }

    if (ComponentForAxis(centroidExtent, splitAxis) < 1e-5f)
    {
        m_bvhNodes[nodeIndex].isLeaf = true;
        return nodeIndex;
    }

    const uint32_t mid = begin + instanceCount / 2u;
    std::nth_element(m_bvhInstanceIndices.begin() + begin,
                     m_bvhInstanceIndices.begin() + mid,
                     m_bvhInstanceIndices.begin() + end,
                     [&](uint32_t lhsIndex, uint32_t rhsIndex)
                     {
                         const float lhs = ComponentForAxis(m_sceneInstances[lhsIndex].worldCenter, splitAxis);
                         const float rhs = ComponentForAxis(m_sceneInstances[rhsIndex].worldCenter, splitAxis);
                         return lhs < rhs;
                     });

    m_bvhNodes[nodeIndex].leftChild = BuildSceneBVHNode(begin, mid);
    m_bvhNodes[nodeIndex].rightChild = BuildSceneBVHNode(mid, end);
    return nodeIndex;
}

void MetalRenderer::CreateDeviceAndSwapchain()
{
    m_device = MTLCreateSystemDefaultDevice();
    m_queue = [m_device newCommandQueue];

    m_view.device = m_device;

    m_view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    m_view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;

    m_view.paused = YES;
    m_view.enableSetNeedsDisplay = NO;
    
}

void MetalRenderer::CreateDepth()
{
    MTLDepthStencilDescriptor* ds = [MTLDepthStencilDescriptor new];
    ds.depthCompareFunction = MTLCompareFunctionLess;
    ds.depthWriteEnabled = YES;
    m_dss = [m_device newDepthStencilStateWithDescriptor:ds];
}

void MetalRenderer::CreateShadersAndPSO()
{
    NSError* err = nil;

    id<MTLLibrary> lib = [m_device newDefaultLibrary];

    id<MTLFunction> vs = [lib newFunctionWithName:@"vs_gbuffer"];
    id<MTLFunction> ps = [lib newFunctionWithName:@"ps_gbuffer"];

    MTLRenderPipelineDescriptor* psoDesc = [MTLRenderPipelineDescriptor new];
    psoDesc.vertexFunction = vs;
    psoDesc.fragmentFunction = ps;
    psoDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    psoDesc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA16Float;
    psoDesc.colorAttachments[2].pixelFormat = MTLPixelFormatRGBA16Float;
    psoDesc.colorAttachments[3].pixelFormat = MTLPixelFormatRGBA16Float;
    psoDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

    MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];

    vd.attributes[0].format = MTLVertexFormatFloat3;
    vd.attributes[0].offset = 0;
    vd.attributes[0].bufferIndex = 0;

    vd.attributes[1].format = MTLVertexFormatFloat3;
    vd.attributes[1].offset = 12;
    
    vd.attributes[2].format = MTLVertexFormatFloat2;
    vd.attributes[2].offset = 24;
    vd.attributes[2].bufferIndex = 0;

    vd.layouts[0].stride = 32;
    vd.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    psoDesc.vertexDescriptor = vd;

    m_gbufferPSO = [m_device newRenderPipelineStateWithDescriptor:psoDesc error:&err];
    if (!m_gbufferPSO) { NSLog(@"GBuffer PSO error: %@", err); }

    id<MTLFunction> vsLighting = [lib newFunctionWithName:@"vs_fullscreen"];
    id<MTLFunction> psLighting = [lib newFunctionWithName:@"ps_lighting"];

    MTLRenderPipelineDescriptor* lightingDesc = [MTLRenderPipelineDescriptor new];
    lightingDesc.vertexFunction = vsLighting;
    lightingDesc.fragmentFunction = psLighting;
    lightingDesc.colorAttachments[0].pixelFormat = m_view.colorPixelFormat;
    lightingDesc.depthAttachmentPixelFormat = m_view.depthStencilPixelFormat;

    m_lightingPSO = [m_device newRenderPipelineStateWithDescriptor:lightingDesc error:&err];
    if (!m_lightingPSO) { NSLog(@"Lighting PSO error: %@", err); }
}

void MetalRenderer::CreateConstantBuffer()
{
    m_cameraCB = [m_device newBufferWithLength:sizeof(CameraCB)
                                       options:MTLResourceStorageModeShared];
}


void MetalRenderer::LoadObjMesh()
{
    char cwd[2048];
    getcwd(cwd, sizeof(cwd));
    NSLog(@"CWD = %s", cwd);

    const std::vector<std::string> objPaths = ResolveModelAssetPaths();
    if (objPaths.empty())
    {
        NSLog(@"No model OBJ files found. Expected names like model.obj, model2.obj, model3.obj in assets.");
        m_indexCount = 0;
        return;
    }

    m_indexCount = 0;
    m_cpuVertices.clear();
    m_cpuIndices.clear();
    m_collisionTriangles.clear();
    m_modelBounds.clear();
    m_sceneInstances.clear();
    m_bvhNodes.clear();
    m_bvhInstanceIndices.clear();
    m_visibleInstances.clear();

    m_batches.clear();
    m_materials.clear();
    m_diffuseTextures.clear();
    m_normalTextures.clear();
    m_heightTextures.clear();
    m_model4PlaneStates.clear();
    bool haveBounds = false;

    for (size_t modelIndex = 0; modelIndex < objPaths.size(); ++modelIndex)
    {
        const std::string& objPath = objPaths[modelIndex];
        ObjMesh mesh;
        NSLog(@"Loading OBJ from: %s", objPath.c_str());
        const bool ok = ObjLoader::LoadMesh(objPath, mesh);

        if (!ok || mesh.vertices.empty() || mesh.indices.empty())
        {
            NSLog(@"OBJ load failed OR empty mesh. path=%s vertices=%lu indices=%lu",
                  objPath.c_str(),
                  (unsigned long)mesh.vertices.size(),
                  (unsigned long)mesh.indices.size());
            continue;
        }

        const uint32_t vertexBase = (uint32_t)m_cpuVertices.size();
        const uint32_t indexBase = (uint32_t)m_cpuIndices.size();
        const uint32_t materialBase = (uint32_t)m_materials.size();
        simd::float3 modelAabbMin = simd::float3{mesh.vertices[0].px, mesh.vertices[0].py, mesh.vertices[0].pz};
        simd::float3 modelAabbMax = modelAabbMin;

        if (!haveBounds)
        {
            m_meshAabbMin = simd::float3{mesh.vertices[0].px, mesh.vertices[0].py, mesh.vertices[0].pz};
            m_meshAabbMax = m_meshAabbMin;
            haveBounds = true;
        }

        for (const VertexPNT& v : mesh.vertices)
        {
            m_cpuVertices.push_back(v);
            if (v.px < modelAabbMin.x) modelAabbMin.x = v.px;
            if (v.py < modelAabbMin.y) modelAabbMin.y = v.py;
            if (v.pz < modelAabbMin.z) modelAabbMin.z = v.pz;
            if (v.px > modelAabbMax.x) modelAabbMax.x = v.px;
            if (v.py > modelAabbMax.y) modelAabbMax.y = v.py;
            if (v.pz > modelAabbMax.z) modelAabbMax.z = v.pz;
            if (v.px < m_meshAabbMin.x) m_meshAabbMin.x = v.px;
            if (v.py < m_meshAabbMin.y) m_meshAabbMin.y = v.py;
            if (v.pz < m_meshAabbMin.z) m_meshAabbMin.z = v.pz;
            if (v.px > m_meshAabbMax.x) m_meshAabbMax.x = v.px;
            if (v.py > m_meshAabbMax.y) m_meshAabbMax.y = v.py;
            if (v.pz > m_meshAabbMax.z) m_meshAabbMax.z = v.pz;
        }

        m_cpuIndices.reserve(m_cpuIndices.size() + mesh.indices.size());
        for (uint32_t idx : mesh.indices)
        {
            m_cpuIndices.push_back(vertexBase + idx);
        }

        m_collisionTriangles.reserve(m_collisionTriangles.size() + mesh.indices.size() / 3);
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3)
        {
            const VertexPNT& va = mesh.vertices[mesh.indices[i + 0]];
            const VertexPNT& vb = mesh.vertices[mesh.indices[i + 1]];
            const VertexPNT& vc = mesh.vertices[mesh.indices[i + 2]];
            const simd::float3 a = simd::float3{va.px, va.py, va.pz};
            const simd::float3 b = simd::float3{vb.px, vb.py, vb.pz};
            const simd::float3 c = simd::float3{vc.px, vc.py, vc.pz};
            const simd::float3 normal = simd::cross(b - a, c - a);
            
            if (simd::length_squared(normal) < 1e-8f)
            {
                continue;
            }

            CollisionTriangle tri;
            tri.a = a;
            tri.b = b;
            tri.c = c;
            tri.normal = simd::normalize(normal);
            if (tri.normal.y <= 0.15f)
            {
                continue;
            }

            tri.aabbMin = Min3(a, b, c);
            tri.aabbMax = Max3(a, b, c);
            m_collisionTriangles.push_back(tri);
        }

        m_materials.reserve(m_materials.size() + mesh.materials.size());
        m_diffuseTextures.reserve(m_diffuseTextures.size() + mesh.materials.size());
        m_normalTextures.reserve(m_normalTextures.size() + mesh.materials.size());
        m_heightTextures.reserve(m_heightTextures.size() + mesh.materials.size());
        for (const ObjMaterial& m : mesh.materials)
        {
            MaterialGPU gpuMat;
            gpuMat.kd_ns = simd::float4{m.kd[0], m.kd[1], m.kd[2], (m.ns > 0.0f) ? m.ns : 32.0f};
            gpuMat.ks_alpha = simd::float4{m.ks[0], m.ks[1], m.ks[2], m.d};
            gpuMat.uvScale = m_textureTiling;
            gpuMat.uvSpeed = m_textureScrollSpeed;
            gpuMat.detailParams = simd::float4{m_meshRadius * m_tessellationStrength, 1.0f, 0.0f, 0.0f};

            id<MTLTexture> diffuseTex = nil;
            if (!m.diffuseTexPath.empty())
            {
                diffuseTex = LoadTextureOrNil(m.diffuseTexPath, true);
            }
            id<MTLTexture> normalTex = nil;
            if (!m.normalTexPath.empty())
            {
                normalTex = LoadTextureOrNil(m.normalTexPath, false);
            }
            id<MTLTexture> heightTex = nil;
            if (!m.heightTexPath.empty())
            {
                heightTex = LoadTextureOrNil(m.heightTexPath, false);
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

            m_materials.push_back(gpuMat);
            m_diffuseTextures.push_back(diffuseTex);
            m_normalTextures.push_back(normalTex);
            m_heightTextures.push_back(heightTex);
        }

        const uint32_t sourceModelIndex = (uint32_t)modelIndex;
        if (m_modelBounds.size() <= sourceModelIndex)
        {
            m_modelBounds.resize(sourceModelIndex + 1u);
        }
        ModelBounds modelBounds;
        modelBounds.localAabbMin = modelAabbMin;
        modelBounds.localAabbMax = modelAabbMax;
        modelBounds.localCenter = (modelAabbMin + modelAabbMax) * 0.5f;
        modelBounds.localRadius = simd::length((modelAabbMax - modelAabbMin) * 0.5f);
        if (modelBounds.localRadius < 1.0f)
        {
            modelBounds.localRadius = 1.0f;
        }
        m_modelBounds[sourceModelIndex] = modelBounds;
        const simd::float3 baseOffset = GetOffsetForModel(sourceModelIndex);
        const bool isModel4 = (sourceModelIndex == 3u);
        const int instanceCount = isModel4 ? m_model4InstanceCount : 1;
        const int gridSize = 100;
        const float spacing = 12.0f;

        m_batches.reserve(m_batches.size() + mesh.submeshes.size() * (size_t)instanceCount);
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
            instance.scale = isModel4 ? 5.0f : 1.0f;
            instance.worldAabbMin = instanceOffset + modelBounds.localAabbMin * instance.scale;
            instance.worldAabbMax = instanceOffset + modelBounds.localAabbMax * instance.scale;
            instance.worldCenter = instanceOffset + modelBounds.localCenter * instance.scale;
            instance.worldRadius = modelBounds.localRadius * instance.scale;
            const uint32_t sceneInstanceIndex = (uint32_t)m_sceneInstances.size();
            m_sceneInstances.push_back(instance);
            m_model4PlaneStates.push_back(0u);

            for (const ObjSubmesh& sm : mesh.submeshes)
            {
                DrawBatch b;
                b.indexOffset = indexBase + sm.indexOffset;
                b.indexCount = sm.indexCount;
                b.materialIndex = materialBase + sm.materialIndex;
                b.sourceModelIndex = sourceModelIndex;
                b.instanceIndex = sceneInstanceIndex;
                m_batches.push_back(b);
            }
        }
    }

    m_indexCount = (uint32_t)m_cpuIndices.size();
    if (m_indexCount == 0 || m_cpuVertices.empty())
    {
        NSLog(@"No valid OBJ meshes were loaded.");
        return;
    }

    m_meshCenter = (m_meshAabbMin + m_meshAabbMax) * 0.5f;
    const simd::float3 extents = (m_meshAabbMax - m_meshAabbMin) * 0.5f;
    m_meshRadius = simd::length(extents);
    if (m_meshRadius < 1.0f)
    {
        m_meshRadius = 1.0f;
    }

    BuildSceneBVH();

    m_vb = [m_device newBufferWithBytes:m_cpuVertices.data()
                                 length:m_cpuVertices.size() * sizeof(VertexPNT)
                                options:MTLResourceStorageModeShared];

    m_ib = [m_device newBufferWithBytes:m_cpuIndices.data()
                                 length:m_cpuIndices.size() * sizeof(uint32_t)
                                options:MTLResourceStorageModeShared];
}

void MetalRenderer::CreateModel4PlaneResources()
{
    const std::string model4TexturePath = ResolveAssetPath("model4.png");
    ObjMesh planeMesh = CreateTexturedPlaneMesh(model4TexturePath);
    m_model4PlaneIndexCount = (uint32_t)planeMesh.indices.size();
    if (!planeMesh.vertices.empty())
    {
        m_model4PlaneVB = [m_device newBufferWithBytes:planeMesh.vertices.data()
                                                length:planeMesh.vertices.size() * sizeof(VertexPNT)
                                               options:MTLResourceStorageModeShared];
    }
    if (!planeMesh.indices.empty())
    {
        m_model4PlaneIB = [m_device newBufferWithBytes:planeMesh.indices.data()
                                                length:planeMesh.indices.size() * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];
    }

    m_model4PlaneMaterial = {};
    m_model4PlaneMaterial.kd_ns = simd::float4{1.0f, 1.0f, 1.0f, 32.0f};
    m_model4PlaneMaterial.ks_alpha = simd::float4{0.0f, 0.0f, 0.0f, 1.0f};
    m_model4PlaneMaterial.uvScale = simd::float2{1.0f, 1.0f};
    m_model4PlaneMaterial.uvSpeed = simd::float2{0.0f, 0.0f};
    m_model4PlaneMaterial.textureFlags = simd::uint4{1u, 0u, 0u, 1u};
    m_model4PlaneMaterial.detailParams = simd::float4{0.0f, 1.0f, 0.0f, 0.0f};
    m_model4PlaneTexture = model4TexturePath.empty() ? nil : LoadTextureOrNil(model4TexturePath, true);
}

id<MTLTexture> MetalRenderer::LoadTextureOrNil(const std::string& path, bool srgb)
{
    NSError* err = nil;
    MTKTextureLoader* loader = [[MTKTextureLoader alloc] initWithDevice:m_device];
    NSDictionary* options = @{
        MTKTextureLoaderOptionSRGB : @(srgb),
        MTKTextureLoaderOptionGenerateMipmaps : @YES
    };
    NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
    NSURL* textureURL = [NSURL fileURLWithPath:nsPath];
    id<MTLTexture> tex = [loader newTextureWithContentsOfURL:textureURL options:options error:&err];
    if (!tex)
    {
        NSLog(@"Texture load failed (%@): %@", nsPath, err);
    }
    return tex;
}

void MetalRenderer::CreateSamplerAndFallbackTexture()
{
    MTLSamplerDescriptor* smpDesc = [MTLSamplerDescriptor new];
    smpDesc.minFilter = MTLSamplerMinMagFilterLinear;
    smpDesc.magFilter = MTLSamplerMinMagFilterLinear;
    smpDesc.mipFilter = MTLSamplerMipFilterLinear;
    smpDesc.sAddressMode = MTLSamplerAddressModeRepeat;
    smpDesc.tAddressMode = MTLSamplerAddressModeRepeat;
    m_sampler = [m_device newSamplerStateWithDescriptor:smpDesc];

    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                   width:1
                                                                                  height:1
                                                                               mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    m_whiteTex = [m_device newTextureWithDescriptor:td];
    m_blackTex = [m_device newTextureWithDescriptor:td];
    m_flatNormalTex = [m_device newTextureWithDescriptor:td];
    uint32_t pixel = 0xffffffffu;
    uint32_t blackPixel = 0x000000ffu;
    uint32_t flatNormalPixel = 0x8080ffffu;
    MTLRegion region = MTLRegionMake2D(0, 0, 1, 1);
    [m_whiteTex replaceRegion:region mipmapLevel:0 withBytes:&pixel bytesPerRow:4];
    [m_blackTex replaceRegion:region mipmapLevel:0 withBytes:&blackPixel bytesPerRow:4];
    [m_flatNormalTex replaceRegion:region mipmapLevel:0 withBytes:&flatNormalPixel bytesPerRow:4];
}

static simd::float4x4 PerspectiveRH(float fovyRadians, float aspect, float zn, float zf)
{
    float ys = 1.0f / tanf(fovyRadians * 0.5f);
    float xs = ys / aspect;
    float zs = zf / (zn - zf);

    simd::float4x4 m{};
    m.columns[0] = { xs, 0,  0,  0 };
    m.columns[1] = { 0,  ys, 0,  0 };
    m.columns[2] = { 0,  0,  zs, -1 };
    m.columns[3] = { 0,  0,  zs * zn, 0 };
    return m;
}

static simd::float4x4 LookAtRH(simd::float3 eye, simd::float3 at, simd::float3 up)
{
    simd::float3 z = simd::normalize(eye - at);
    simd::float3 x = simd::normalize(simd::cross(up, z));
    simd::float3 y = simd::cross(z, x);

    simd::float4x4 m = matrix_identity_float4x4;
    m.columns[0] = { x.x, y.x, z.x, 0 };
    m.columns[1] = { x.y, y.y, z.y, 0 };
    m.columns[2] = { x.z, y.z, z.z, 0 };
    m.columns[3] = { -simd::dot(x, eye), -simd::dot(y, eye), -simd::dot(z, eye), 1 };
    return m;
}

static simd::float4x4 RotationY(float a)
{
    float c = cosf(a), s = sinf(a);
    simd::float4x4 m = matrix_identity_float4x4;
    m.columns[0] = { c, 0, -s, 0 };
    m.columns[2] = { s, 0,  c, 0 };
    return m;
}

void MetalRenderer::DrawFrame()
{
    if (m_indexCount == 0 || !m_vb || !m_ib) {
        //очистка кадра
        return;
    }
    @autoreleasepool
    {
        MTLRenderPassDescriptor* rp = m_view.currentRenderPassDescriptor;
        id<CAMetalDrawable> drawable = m_view.currentDrawable;
        if (!rp || !drawable) return;

        const NSUInteger drawableWidth = (NSUInteger)m_view.drawableSize.width;
        const NSUInteger drawableHeight = (NSUInteger)m_view.drawableSize.height;
        m_gbuffer.EnsureTextures(drawableWidth, drawableHeight);
        if (!m_gbuffer.IsReady())
        {
            return;
        }

        MTLRenderPassDescriptor* gbufferPass = m_gbuffer.CreateRenderPassDescriptor();
        if (!gbufferPass)
        {
            return;
        }

        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0.5, 0.7, 1.0, 1.0);
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;

        CameraCB* cb = (CameraCB*)m_cameraCB.contents;

        const float dt = 0.016f;

        //без анимации, статическая матрица
        cb->world = matrix_identity_float4x4;

        //управление камерой
        InputDevice& inp = InputDevice::Get();
        const int kRMB = 1;  //правая кнопка мыши (macOS: 0=left, 1=right)
        if (inp.MouseButtonHeld(kRMB)) {
            float dx = inp.MouseDeltaX();
            float dy = inp.MouseDeltaY();
            
            if (fabsf(dx) < 100.0f && fabsf(dy) < 100.0f)
            {
                m_yaw   -= dx * m_mouseSens;
                m_pitch -= dy * m_mouseSens;
                const float maxPitch = (float)(M_PI * 0.5) - 0.01f;
                if (m_pitch > maxPitch)  m_pitch = maxPitch;
                if (m_pitch < -maxPitch) m_pitch = -maxPitch;
            }
        }

        //направление взгляда (yaw/pitch)
        simd::float3 front;
        front.x = sinf(m_yaw) * cosf(m_pitch);
        front.y = sinf(m_pitch);
        front.z = -cosf(m_yaw) * cosf(m_pitch);
        front = simd::normalize(front);
        //горизонтальное направление вперёд для WASD
        simd::float3 frontXZ = simd::float3{ sinf(m_yaw), 0.0f, -cosf(m_yaw) };
        simd::float3 right   = simd::float3{ cosf(m_yaw), 0.0f, sinf(m_yaw) };

        //W=13, S=1, A=0, D=2, Space=49, Shift=56/60
        if (inp.KeyHeld(13))  m_camPos += frontXZ * (m_camSpeed * dt);  // W
        if (inp.KeyHeld(1))   m_camPos -= frontXZ * (m_camSpeed * dt); // S
        if (inp.KeyHeld(0))   m_camPos -= right * (m_camSpeed * dt);   // A
        if (inp.KeyHeld(2))   m_camPos += right * (m_camSpeed * dt);   // D
        if (inp.KeyHeld(49))  m_camPos.y += m_camSpeed * dt;           // Space
        if (inp.ModifierShift()) m_camPos.y -= m_camSpeed * dt;        // Shift — вниз
        if (inp.KeyPressed(17)) // T
        {
            m_enableFrustumCulling = !m_enableFrustumCulling;
            NSLog(@"Frustum culling %@", m_enableFrustumCulling ? @"enabled" : @"disabled");
        }
        if (inp.KeyPressed(16)) // Y
        {
            m_enableBvhFrustumCulling = !m_enableBvhFrustumCulling;
            NSLog(@"BVH frustum culling %@", m_enableBvhFrustumCulling ? @"enabled" : @"disabled");
        }

        simd::float3 target = m_camPos + front;
        cb->view = LookAtRH(m_camPos, target, simd::float3{0, 1, 0});
        cb->cameraPos = m_camPos;
        m_timeSeconds += dt;

        const float enterPlaneDistance = m_model4PlaneSwapDistance;
        const float exitPlaneDistance = m_model4PlaneSwapDistance - m_model4PlaneSwapHysteresis;
        for (uint32_t instanceIndex = 0; instanceIndex < m_sceneInstances.size(); ++instanceIndex)
        {
            const SceneInstance& instance = m_sceneInstances[instanceIndex];
            if (instance.sourceModelIndex != 3u)
            {
                continue;
            }

            const float distance = simd::distance(m_camPos, instance.worldCenter);
            if (instanceIndex >= m_model4PlaneStates.size())
            {
                continue;
            }

            if (!m_model4PlaneStates[instanceIndex] && distance > enterPlaneDistance)
            {
                m_model4PlaneStates[instanceIndex] = 1u;
            }
            else if (m_model4PlaneStates[instanceIndex] && distance < exitPlaneDistance)
            {
                m_model4PlaneStates[instanceIndex] = 0u;
            }
        }

        // Texture animation is disabled for now: keep UV scroll time fixed at zero.
        cb->timeSeconds = 0.0f;

        // projection по размеру окна
        float w = (float)drawableWidth;
        float h = (float)drawableHeight;
        float aspect = (h > 0.0f) ? (w / h) : 1.0f;

        const float nearPlane = (m_meshRadius > 50.0f) ? 1.0f : 0.1f;
        const float farPlane = fmaxf(nearPlane + 1.0f, m_meshRadius * 6.0f);
        cb->proj = PerspectiveRH(60.0f * (float)M_PI / 180.0f, aspect, nearPlane, farPlane);

        cb->lightDir = m_directionalLight.GetDirection();
        cb->lightIntensity = m_directionalLight.GetIntensity();
        cb->lightColor = m_directionalLight.GetColor();

        const float cameraToMeshDistance = simd::distance(m_camPos, m_meshCenter);
        const float tessellationFadeNear =
            fmaxf(m_meshRadius * m_tessellationFadeNearMultiplier, 0.0f);
        const float tessellationFadeFar =
            fmaxf(tessellationFadeNear + 0.001f, m_meshRadius * m_tessellationFadeFarMultiplier);
        float tessellationFactor =
            1.0f - ((cameraToMeshDistance - tessellationFadeNear) /
                    (tessellationFadeFar - tessellationFadeNear));
        tessellationFactor = fmaxf(0.0f, fminf(tessellationFactor, 1.0f));
        const float tanHalfFovY = tanf(60.0f * (float)M_PI / 360.0f);
        const float tanHalfFovX = tanHalfFovY * aspect;

        id<MTLCommandBuffer> cmd = [m_queue commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:gbufferPass];

        // Geometry pass: fill the G-buffer.
        [enc setRenderPipelineState:m_gbufferPSO];
        [enc setDepthStencilState:m_dss];

        [enc setVertexBuffer:m_vb offset:0 atIndex:0];
        [enc setFragmentBuffer:m_cameraCB offset:0 atIndex:0];
        [enc setFragmentSamplerState:m_sampler atIndex:0];
        [enc setVertexSamplerState:m_sampler atIndex:0];

        if (!m_visibleInstances.empty())
        {
            std::fill(m_visibleInstances.begin(), m_visibleInstances.end(), 0u);
        }

        if (!m_enableFrustumCulling)
        {
            std::fill(m_visibleInstances.begin(), m_visibleInstances.end(), 1u);
        }
        else if (m_enableBvhFrustumCulling && !m_bvhNodes.empty())
        {
            std::vector<uint32_t> nodeStack;
            nodeStack.push_back(0u);
            while (!nodeStack.empty())
            {
                const uint32_t nodeIndex = nodeStack.back();
                nodeStack.pop_back();

                const BvhNode& node = m_bvhNodes[nodeIndex];
                const simd::float3 nodeCenter = (node.aabbMin + node.aabbMax) * 0.5f;
                const float nodeRadius = simd::length((node.aabbMax - node.aabbMin) * 0.5f);
                if (!IsSphereVisibleInFrustum(cb->view,
                                              nodeCenter,
                                              nodeRadius,
                                              nearPlane,
                                              farPlane,
                                              tanHalfFovX,
                                              tanHalfFovY))
                {
                    continue;
                }

                if (node.isLeaf)
                {
                    for (uint32_t i = 0; i < node.instanceCount; ++i)
                    {
                        const uint32_t instanceIndex = m_bvhInstanceIndices[node.firstInstance + i];
                        const SceneInstance& instance = m_sceneInstances[instanceIndex];
                        if (IsSphereVisibleInFrustum(cb->view,
                                                     instance.worldCenter,
                                                     instance.worldRadius,
                                                     nearPlane,
                                                     farPlane,
                                                     tanHalfFovX,
                                                     tanHalfFovY))
                        {
                            m_visibleInstances[instanceIndex] = 1u;
                        }
                    }
                }
                else
                {
                    if (node.leftChild != UINT32_MAX)
                    {
                        nodeStack.push_back(node.leftChild);
                    }
                    if (node.rightChild != UINT32_MAX)
                    {
                        nodeStack.push_back(node.rightChild);
                    }
                }
            }
        }
        else
        {
            for (uint32_t instanceIndex = 0; instanceIndex < m_sceneInstances.size(); ++instanceIndex)
            {
                const SceneInstance& instance = m_sceneInstances[instanceIndex];
                if (IsSphereVisibleInFrustum(cb->view,
                                             instance.worldCenter,
                                             instance.worldRadius,
                                             nearPlane,
                                             farPlane,
                                             tanHalfFovX,
                                             tanHalfFovY))
                {
                    m_visibleInstances[instanceIndex] = 1u;
                }
            }
        }

        for (const DrawBatch& b : m_batches)
        {
            if (b.instanceIndex >= m_visibleInstances.size() || m_visibleInstances[b.instanceIndex] == 0u)
            {
                continue;
            }

            const SceneInstance& instance = m_sceneInstances[b.instanceIndex];
            const bool usePlaneForInstance =
                instance.sourceModelIndex == 3u &&
                b.instanceIndex < m_model4PlaneStates.size() &&
                m_model4PlaneStates[b.instanceIndex] != 0u;
            if (usePlaneForInstance)
            {
                continue;
            }
            MaterialGPU mat{};
            if (b.materialIndex < m_materials.size())
            {
                mat = m_materials[b.materialIndex];
            }
            const float modelTessellationStrength = GetTessellationStrengthForModel(b.sourceModelIndex);
            const float displacementStrength =
                m_meshRadius * modelTessellationStrength * tessellationFactor;
            mat.detailParams.x = displacementStrength;

            CameraCB localCb = *cb;
            localCb.world = simd_mul(MatTranslation(instance.worldOffset), MatScale(instance.scale));
            [enc setVertexBytes:&localCb length:sizeof(CameraCB) atIndex:1];
            [enc setVertexBytes:&mat length:sizeof(MaterialGPU) atIndex:2];
            [enc setFragmentBytes:&mat length:sizeof(MaterialGPU) atIndex:1];

            id<MTLTexture> diffuseTex = m_whiteTex;
            if (b.materialIndex < m_diffuseTextures.size() && m_diffuseTextures[b.materialIndex])
            {
                diffuseTex = m_diffuseTextures[b.materialIndex];
            }
            id<MTLTexture> normalTex = m_flatNormalTex;
            if (b.materialIndex < m_normalTextures.size() && m_normalTextures[b.materialIndex])
            {
                normalTex = m_normalTextures[b.materialIndex];
            }
            id<MTLTexture> heightTex = m_blackTex;
            if (b.materialIndex < m_heightTextures.size() && m_heightTextures[b.materialIndex])
            {
                heightTex = m_heightTextures[b.materialIndex];
            }
            [enc setVertexTexture:heightTex atIndex:0];
            [enc setFragmentTexture:diffuseTex atIndex:0];
            [enc setFragmentTexture:normalTex atIndex:1];

            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:b.indexCount
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:m_ib
                     indexBufferOffset:(NSUInteger)b.indexOffset * sizeof(uint32_t)];
        }

        if (m_model4PlaneVB && m_model4PlaneIB && m_model4PlaneIndexCount > 0u)
        {
            [enc setVertexBuffer:m_model4PlaneVB offset:0 atIndex:0];
            [enc setVertexTexture:m_blackTex atIndex:0];
            [enc setFragmentTexture:(m_model4PlaneTexture ? m_model4PlaneTexture : m_whiteTex) atIndex:0];
            [enc setFragmentTexture:m_flatNormalTex atIndex:1];

            for (uint32_t instanceIndex = 0; instanceIndex < m_sceneInstances.size(); ++instanceIndex)
            {
                if (instanceIndex >= m_visibleInstances.size() || m_visibleInstances[instanceIndex] == 0u)
                {
                    continue;
                }
                if (instanceIndex >= m_model4PlaneStates.size() || m_model4PlaneStates[instanceIndex] == 0u)
                {
                    continue;
                }

                const SceneInstance& instance = m_sceneInstances[instanceIndex];
                if (instance.sourceModelIndex != 3u)
                {
                    continue;
                }

                CameraCB localCb = *cb;
                localCb.world = simd_mul(simd_mul(MatTranslation(instance.worldOffset),
                                                  MatBillboardFacingCamera(instance.worldOffset, m_camPos)),
                                         MatScale(m_model4PlaneScale));
                [enc setVertexBytes:&localCb length:sizeof(CameraCB) atIndex:1];
                [enc setVertexBytes:&m_model4PlaneMaterial length:sizeof(MaterialGPU) atIndex:2];
                [enc setFragmentBytes:&m_model4PlaneMaterial length:sizeof(MaterialGPU) atIndex:1];

                [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                indexCount:m_model4PlaneIndexCount
                                 indexType:MTLIndexTypeUInt32
                               indexBuffer:m_model4PlaneIB
                         indexBufferOffset:0];
            }

            [enc setVertexBuffer:m_vb offset:0 atIndex:0];
        }

        [enc endEncoding];

        id<MTLRenderCommandEncoder> lightingEnc = [cmd renderCommandEncoderWithDescriptor:rp];
        if (!lightingEnc || !m_lightingPSO)
        {
            [cmd presentDrawable:drawable];
            [cmd commit];
            return;
        }
        [lightingEnc setRenderPipelineState:m_lightingPSO];
        [lightingEnc setFragmentBuffer:m_cameraCB offset:0 atIndex:0];
        [lightingEnc setFragmentSamplerState:m_sampler atIndex:0];
        [lightingEnc setFragmentTexture:m_gbuffer.GetAlbedoTexture() atIndex:0];
        [lightingEnc setFragmentTexture:m_gbuffer.GetNormalTexture() atIndex:1];
        [lightingEnc setFragmentTexture:m_gbuffer.GetPositionTexture() atIndex:2];
        [lightingEnc setFragmentTexture:m_gbuffer.GetMaterialTexture() atIndex:3];
        [lightingEnc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [lightingEnc endEncoding];

        [cmd presentDrawable:drawable];
        [cmd commit];
    }
}
