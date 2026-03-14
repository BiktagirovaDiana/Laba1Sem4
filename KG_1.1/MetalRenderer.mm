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
#include <cmath>
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
    std::vector<std::string> candidateDirs;

    // Prefer assets located next to the checked-in source tree over stale build artifacts.
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

    for (const std::string& d : candidateDirs)
    {
        const std::string full = JoinPath(d, fileName);
        if (FileExists(full.c_str()))
        {
            return full;
        }
    }

    return std::string();
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
    simd::float3   pointLightPos;
    float          pointLightRange;
    simd::float3   pointLightColor;
    float          pointLightIntensity;
    simd::float3   spotLightPos;
    float          spotLightRange;
    simd::float3   spotLightDir;
    float          spotLightConeAngle;
    simd::float3   spotLightColor;
    float          spotLightIntensity;
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
    LoadObjMesh();
    m_directionalLight = DirectionalLight(simd::float3{-0.3f, -1.0f, -0.2f}, // направление
                                          simd::float3{0.2f, 0.4f, 1.0f}, // цвет
                                          //simd::float3{1.0f, 1.0f, 1.0f},
                                          1.5f); // интенсивность
    
    m_pointLight = PointLight(simd::float3{0.0f, 200.0f, 0.0f}, // позиция
                              500.0f, // радиус
                              simd::float3{0.2f, 1.0f, 0.2f}, // цвет
                              10.0f); // интенсивность
    m_thrownLightPos = m_pointLight.GetPosition();
    m_spotLight = SpotLight(simd::float3{0.0f, 40.0f, 400.0f}, // позиция
                            simd::float3{0.0f, -1.0f, -1.0f}, // направление
                            100, // радиус
                            60.0f, // угол конуса в градусах
                            simd::float3{1.0f, 0.1f, 0.1f}, // красный цвет
                            40.0f); // интенсивность
}
 
MetalRenderer::~MetalRenderer() {}

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
    ObjMesh mesh;

    char cwd[2048];
    getcwd(cwd, sizeof(cwd));
    NSLog(@"CWD = %s", cwd);

    const std::string objPath = ResolveAssetPath("model.obj");
    if (objPath.empty())
    {
        NSLog(@"OBJ not found. Tried: assets/model.obj, KG_1.1/assets/model.obj, executable-relative paths");
        m_indexCount = 0;
        return;
    }

    const std::string assetsDir = DirName(objPath);
    NSLog(@"Loading OBJ from: %s", objPath.c_str());

    const bool ok = ObjLoader::LoadMesh(objPath, mesh);
    
    if (!ok || mesh.vertices.empty() || mesh.indices.empty())
    {
        NSLog(@"OBJ load failed OR empty mesh. vertices=%lu indices=%lu",
              (unsigned long)mesh.vertices.size(), (unsigned long)mesh.indices.size());
        m_indexCount = 0;
        return;
    }

    m_indexCount = (uint32_t)mesh.indices.size();
    m_cpuVertices = mesh.vertices;
    m_cpuIndices = mesh.indices;
    m_collisionTriangles.clear();
    m_collisionTriangles.reserve(mesh.indices.size() / 3);

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

    // Bounds in object space are used to make texture animation speed depend on camera distance.
    
    m_meshAabbMin = simd::float3{mesh.vertices[0].px, mesh.vertices[0].py, mesh.vertices[0].pz};
    m_meshAabbMax = m_meshAabbMin;
    for (const VertexPNT& v : mesh.vertices)
    {
        if (v.px < m_meshAabbMin.x) m_meshAabbMin.x = v.px;
        if (v.py < m_meshAabbMin.y) m_meshAabbMin.y = v.py;
        if (v.pz < m_meshAabbMin.z) m_meshAabbMin.z = v.pz;
        if (v.px > m_meshAabbMax.x) m_meshAabbMax.x = v.px;
        if (v.py > m_meshAabbMax.y) m_meshAabbMax.y = v.py;
        if (v.pz > m_meshAabbMax.z) m_meshAabbMax.z = v.pz;
    }

    m_meshCenter = (m_meshAabbMin + m_meshAabbMax) * 0.5f;
    const simd::float3 extents = (m_meshAabbMax - m_meshAabbMin) * 0.5f;
    m_meshRadius = simd::length(extents);
    if (m_meshRadius < 1.0f)
    {
        m_meshRadius = 1.0f;
    }

    // Keep the original fixed camera start.
    m_camPos = simd::float3{0.0f, 0.0f, 3.0f};
    m_camSpeed = 120.0f;
    m_yaw = (float)M_PI;
    m_pitch = 0.0f;

    m_vb = [m_device newBufferWithBytes:mesh.vertices.data()
                                 length:mesh.vertices.size() * sizeof(VertexPNT)
                                options:MTLResourceStorageModeShared];

    m_ib = [m_device newBufferWithBytes:mesh.indices.data()
                                 length:mesh.indices.size() * sizeof(uint32_t)
                                options:MTLResourceStorageModeShared];

    m_batches.clear();
    m_materials.clear();
    m_materialTextures.clear();
    m_batches.reserve(mesh.submeshes.size());
    m_materials.reserve(mesh.materials.size());
    m_materialTextures.reserve(mesh.materials.size());

    for (const ObjMaterial& m : mesh.materials)
    {
        MaterialGPU gpuMat;
        gpuMat.kd_ns = simd::float4{m.kd[0], m.kd[1], m.kd[2], (m.ns > 0.0f) ? m.ns : 32.0f};
        gpuMat.ks_alpha = simd::float4{m.ks[0], m.ks[1], m.ks[2], m.d};
        gpuMat.uvScale = m_textureTiling;
        gpuMat.uvSpeed = m_textureScrollSpeed;

        id<MTLTexture> tex = nil;
        if (!m.diffuseTexPath.empty())
        {
            tex = LoadTextureOrNil(m.diffuseTexPath);
        }
        if (tex)
        {
            gpuMat.useTexture = 1;
        }

        NSLog(@"Material '%s': diffuseTex=%s",
              m.name.empty() ? "<default>" : m.name.c_str(),
              m.diffuseTexPath.empty() ? "<none>" : m.diffuseTexPath.c_str());

        m_materials.push_back(gpuMat);
        m_materialTextures.push_back(tex);
    }

    for (const ObjSubmesh& sm : mesh.submeshes)
    {
        DrawBatch b;
        b.indexOffset = sm.indexOffset;
        b.indexCount = sm.indexCount;
        b.materialIndex = sm.materialIndex;
        m_batches.push_back(b);
    }
}

id<MTLTexture> MetalRenderer::LoadTextureOrNil(const std::string& path)
{
    NSError* err = nil;
    MTKTextureLoader* loader = [[MTKTextureLoader alloc] initWithDevice:m_device];
    NSDictionary* options = @{
        MTKTextureLoaderOptionSRGB : @YES,
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
    uint32_t pixel = 0xffffffffu;
    MTLRegion region = MTLRegionMake2D(0, 0, 1, 1);
    [m_whiteTex replaceRegion:region mipmapLevel:0 withBytes:&pixel bytesPerRow:4];
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

static bool SegmentTriangleIntersection(simd::float3 start,
                                        simd::float3 end,
                                        simd::float3 a,
                                        simd::float3 b,
                                        simd::float3 c,
                                        float& outT,
                                        simd::float3& outPoint,
                                        simd::float3& outNormal)
{
    const simd::float3 dir = end - start;
    const simd::float3 edge1 = b - a;
    const simd::float3 edge2 = c - a;
    const simd::float3 p = simd::cross(dir, edge2);
    const float det = simd::dot(edge1, p);
    if (fabsf(det) < 1e-6f)
    {
        return false;
    }

    const float invDet = 1.0f / det;
    const simd::float3 tVec = start - a;
    const float u = simd::dot(tVec, p) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }

    const simd::float3 q = simd::cross(tVec, edge1);
    const float v = simd::dot(dir, q) * invDet;
    if (v < 0.0f || (u + v) > 1.0f)
    {
        return false;
    }

    const float t = simd::dot(edge2, q) * invDet;
    if (t < 0.0f || t > 1.0f)
    {
        return false;
    }

    const simd::float3 normal = simd::cross(edge1, edge2);
    if (simd::length_squared(normal) < 1e-8f)
    {
        return false;
    }

    outT = t;
    outPoint = start + dir * t;
    outNormal = simd::normalize(normal);
    return true;
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

        //W=13, S=1, A=0, D=2, E=14, Space=49, Shift=56/60
        if (inp.KeyHeld(13))  m_camPos += frontXZ * (m_camSpeed * dt);  // W
        if (inp.KeyHeld(1))   m_camPos -= frontXZ * (m_camSpeed * dt); // S
        if (inp.KeyHeld(0))   m_camPos -= right * (m_camSpeed * dt);   // A
        if (inp.KeyHeld(2))   m_camPos += right * (m_camSpeed * dt);   // D
        if (inp.KeyHeld(49))  m_camPos.y += m_camSpeed * dt;           // Space
        if (inp.ModifierShift()) m_camPos.y -= m_camSpeed * dt;        // Shift — вниз

        if (inp.KeyPressed(14))
        {
            m_thrownLightActive = true;
            m_thrownLightLanded = false;
            m_thrownLightForward = front;
            m_thrownLightPos = m_camPos + front * m_thrownLightSpawnOffset;
            m_thrownLightVelocity = front * m_thrownLightLaunchSpeed;
            m_thrownLightVelocity.y -= m_thrownLightInitialDropSpeed;
        }

        if (m_thrownLightActive && !m_thrownLightLanded)
        {
            const simd::float3 previousPos = m_thrownLightPos;
            m_thrownLightVelocity.y -= m_thrownLightGravity * dt;
            m_thrownLightPos += m_thrownLightVelocity * dt;

            bool hitSurface = false;
            float closestT = 1.0f;
            simd::float3 hitPoint = m_thrownLightPos;
            simd::float3 hitNormal = simd::float3{0.0f, 1.0f, 0.0f};
            const simd::float3 sweepMin = Min3(previousPos, m_thrownLightPos, previousPos);
            const simd::float3 sweepMax = Max3(previousPos, m_thrownLightPos, previousPos);

            for (const CollisionTriangle& tri : m_collisionTriangles)
            {
                if (tri.aabbMax.x < sweepMin.x || tri.aabbMin.x > sweepMax.x ||
                    tri.aabbMax.y < sweepMin.y || tri.aabbMin.y > sweepMax.y ||
                    tri.aabbMax.z < sweepMin.z || tri.aabbMin.z > sweepMax.z)
                {
                    continue;
                }

                float t = 0.0f;
                simd::float3 candidatePoint;
                simd::float3 candidateNormal;
                if (!SegmentTriangleIntersection(previousPos, m_thrownLightPos,
                                                 tri.a, tri.b, tri.c,
                                                 t, candidatePoint, candidateNormal))
                {
                    continue;
                }

                if (!hitSurface || t < closestT)
                {
                    hitSurface = true;
                    closestT = t;
                    hitPoint = candidatePoint;
                    hitNormal = candidateNormal;
                }
            }

            if (hitSurface)
            {
                m_thrownLightPos = hitPoint + hitNormal * m_thrownLightSurfaceOffset;
                m_thrownLightVelocity = simd::float3{0.0f, 0.0f, 0.0f};
                m_thrownLightLanded = true;
            }
            else if (m_thrownLightPos.y <= m_meshAabbMin.y)
            {
                m_thrownLightPos.y = m_meshAabbMin.y;
                m_thrownLightVelocity = simd::float3{0.0f, 0.0f, 0.0f};
                m_thrownLightLanded = true;
            }
        }

        m_pointLight = PointLight(m_thrownLightPos,
                                  m_pointLight.GetRange(),
                                  m_pointLight.GetColor(),
                                  m_pointLight.GetIntensity());

        simd::float3 target = m_camPos + front;
        cb->view = LookAtRH(m_camPos, target, simd::float3{0, 1, 0});
        cb->cameraPos = m_camPos;
        m_timeSeconds += dt;

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
        
        cb->pointLightPos = m_pointLight.GetPosition();
        cb->pointLightRange = m_pointLight.GetRange();
        cb->pointLightColor = m_pointLight.GetColor();
        cb->pointLightIntensity = m_pointLight.GetIntensity();
        cb->spotLightPos = m_spotLight.GetPosition();
        cb->spotLightRange = m_spotLight.GetRange();
        cb->spotLightDir = m_spotLight.GetDirection();
        cb->spotLightConeAngle = m_spotLight.GetConeAngleRadians();
        cb->spotLightColor = m_spotLight.GetColor();
        cb->spotLightIntensity = m_spotLight.GetIntensity();

        id<MTLCommandBuffer> cmd = [m_queue commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:gbufferPass];

        // Geometry pass: fill the G-buffer.
        [enc setRenderPipelineState:m_gbufferPSO];
        [enc setDepthStencilState:m_dss];

        [enc setVertexBuffer:m_vb offset:0 atIndex:0];
        [enc setVertexBuffer:m_cameraCB offset:0 atIndex:1];

        [enc setFragmentBuffer:m_cameraCB offset:0 atIndex:0];
        [enc setFragmentSamplerState:m_sampler atIndex:0];

        for (const DrawBatch& b : m_batches)
        {
            MaterialGPU mat{};
            if (b.materialIndex < m_materials.size())
            {
                mat = m_materials[b.materialIndex];
            }
            [enc setFragmentBytes:&mat length:sizeof(MaterialGPU) atIndex:1];

            id<MTLTexture> tex = m_whiteTex;
            if (b.materialIndex < m_materialTextures.size() && m_materialTextures[b.materialIndex])
            {
                tex = m_materialTextures[b.materialIndex];
            }
            [enc setFragmentTexture:tex atIndex:0];

            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:b.indexCount
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:m_ib
                     indexBufferOffset:(NSUInteger)b.indexOffset * sizeof(uint32_t)];
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
