//
//  MetalRenderer.mm

//  KG_1.1
//
//  Created by Macbook on 21.02.2026.
//

#import "MetalRenderer.hpp"
#include <simd/simd.h>
#include "InputDevice.hpp"
#import <Foundation/Foundation.h>

#include <unistd.h>
#include <sys/stat.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

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

static bool FileExists(const char* p)
{
    struct stat st;
    return (stat(p, &st) == 0) && S_ISREG(st.st_mode);
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

MetalRenderer::MetalRenderer(MTKView* view) : m_view(view)
{
    CreateDeviceAndSwapchain();
    m_gbuffer.SetDevice(m_device);
    CreateDepth();
    CreateShadowResources();
    CreateShadersAndPSO();
    CreateConstantBuffer();
    CreateStructuredBuffers();
    CreateSamplerAndFallbackTexture();
    CreateTerrainResources();
    CreateModelResources();
    CreateParticleResources();
    LoadIrradianceMap();
    LoadBrdfLut();
    LoadPrefilteredMap();
    m_camPos = simd::float3{0.0f, 0.0f, 3.0f};
    m_camSpeed = 120.0f;
    m_yaw = (float)M_PI;
    m_pitch = 0.0f;
    LoadObjMesh();
    m_directionalLight = DirectionalLight(simd::float3{0.0f, -1.0f, 0.0f},
                                          simd::float3{1.0f, 1.0f, 1.0f},
                                          2.5f);
}
 
MetalRenderer::~MetalRenderer() {}

void MetalRenderer::CreateTerrainResources()
{
    m_terrain.CreateResources(m_device,
                              [this](const std::string& path, bool srgb)
                              {
                                  return LoadTextureOrNil(path, srgb);
                              });
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

void MetalRenderer::CreateShadowResources()
{
    m_shadowRenderer.CreateResources(m_device);
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

    m_shadowRenderer.CreatePipelines(m_device, lib, vd);
    m_models.CreatePipelines(m_device, lib, vd, MTLPixelFormatDepth32Float);
    m_particleRenderer.CreatePipelines(m_device, lib, psoDesc);

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

void MetalRenderer::CreateStructuredBuffers()
{
    const NSUInteger bufferLength =
        (NSUInteger)kStructuredBufferCapacity * sizeof(StructuredBufferElement);

    m_appendStructuredBuffer = [m_device newBufferWithLength:bufferLength
                                                     options:MTLResourceStorageModeShared];
    m_consumeStructuredBuffer = [m_device newBufferWithLength:bufferLength
                                                      options:MTLResourceStorageModeShared];
    m_appendCounterBuffer = [m_device newBufferWithLength:sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
    m_consumeCounterBuffer = [m_device newBufferWithLength:sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];

    if (!m_appendStructuredBuffer || !m_consumeStructuredBuffer ||
        !m_appendCounterBuffer || !m_consumeCounterBuffer)
    {
        NSLog(@"Failed to create append/consume structured buffers.");
        return;
    }

    std::memset(m_appendStructuredBuffer.contents, 0, bufferLength);

    StructuredBufferElement* consumeData =
        static_cast<StructuredBufferElement*>(m_consumeStructuredBuffer.contents);
    for (uint32_t i = 0; i < kStructuredBufferCapacity; ++i)
    {
        consumeData[i].value = i;
    }

    *static_cast<uint32_t*>(m_appendCounterBuffer.contents) = 0u;
    *static_cast<uint32_t*>(m_consumeCounterBuffer.contents) = kStructuredBufferCapacity;
}


void MetalRenderer::LoadObjMesh()
{
    const bool ok = m_scene.Load([this](const std::string& path, bool srgb)
                                 {
                                     return LoadTextureOrNil(path, srgb);
                                 });
    if (!ok || m_scene.IndexCount() == 0 || m_scene.Vertices().empty())
    {
        return;
    }

    BuildRainCollisionPlanes();

    m_vb = [m_device newBufferWithBytes:m_scene.Vertices().data()
                                 length:m_scene.Vertices().size() * sizeof(VertexPNT)
                                options:MTLResourceStorageModeShared];

    m_ib = [m_device newBufferWithBytes:m_scene.Indices().data()
                                 length:m_scene.Indices().size() * sizeof(uint32_t)
                                options:MTLResourceStorageModeShared];
}

void MetalRenderer::BuildRainCollisionPlanes()
{
    std::vector<ParticleRenderer::CollisionAabb> collisionAabbs;
    collisionAabbs.reserve(m_scene.CollisionAabbs().size());
    for (const Scene::CollisionAabb& sceneAabb : m_scene.CollisionAabbs())
    {
        ParticleRenderer::CollisionAabb aabb;
        aabb.min = sceneAabb.min;
        aabb.max = sceneAabb.max;
        collisionAabbs.push_back(aabb);
    }

    m_particleRenderer.BuildRainCollisionPlanes(m_device, collisionAabbs);
}

void MetalRenderer::CreateModelResources()
{
    m_models.CreateResources(m_device,
                             [](const std::string& fileName)
                             {
                                 return ResolveAssetPath(fileName);
                             },
                             [this](const std::string& path, bool srgb)
                             {
                                 return LoadTextureOrNil(path, srgb);
                             });
}

void MetalRenderer::CreateParticleResources()
{
    m_particleRenderer.CreateResources(m_device,
                                       [](const std::string& fileName)
                                       {
                                           return ResolveAssetPath(fileName);
                                       },
                                       [this](const std::string& path, bool srgb)
                                       {
                                           return LoadTextureOrNil(path, srgb);
                                       });
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

id<MTLTexture> MetalRenderer::LoadCubeTextureOrNil(const std::string& path, bool srgb)
{
    NSError* err = nil;
    MTKTextureLoader* loader = [[MTKTextureLoader alloc] initWithDevice:m_device];
    NSDictionary* options = @{
        MTKTextureLoaderOptionSRGB : @(srgb),
        MTKTextureLoaderOptionGenerateMipmaps : @NO
    };
    NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
    NSURL* textureURL = [NSURL fileURLWithPath:nsPath];
    id<MTLTexture> tex = [loader newTextureWithContentsOfURL:textureURL options:options error:&err];
    if (!tex)
    {
        NSLog(@"Cube texture load failed (%@): %@", nsPath, err);
        return nil;
    }

    if (tex.textureType != MTLTextureTypeCube)
    {
        NSLog(@"Cube texture load skipped (%@): texture type is %lu, expected cube",
              nsPath,
              (unsigned long)tex.textureType);
        return nil;
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

    m_shadowRenderer.CreateSampler(m_device);

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

    MTLTextureDescriptor* cubeDesc =
        [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                              size:1
                                                         mipmapped:NO];
    cubeDesc.usage = MTLTextureUsageShaderRead;
    m_fallbackIrradianceMap = [m_device newTextureWithDescriptor:cubeDesc];
    const uint32_t irradiancePixel = 0xff242018u;
    for (NSUInteger face = 0; face < 6; ++face)
    {
        [m_fallbackIrradianceMap replaceRegion:region
                                   mipmapLevel:0
                                         slice:face
                                     withBytes:&irradiancePixel
                                   bytesPerRow:4
                                 bytesPerImage:4];
    }

    // Fallback BRDF LUT: 1x1 RG8 with scale=1, bias=0 (f0 * 1 + 0 = f0)
    MTLTextureDescriptor* brdfDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG8Unorm
                                                           width:1
                                                          height:1
                                                       mipmapped:NO];
    brdfDesc.usage = MTLTextureUsageShaderRead;
    m_fallbackBrdfLut = [m_device newTextureWithDescriptor:brdfDesc];
    const uint8_t brdfPixel[2] = { 0xff, 0x00 };
    [m_fallbackBrdfLut replaceRegion:region mipmapLevel:0 withBytes:brdfPixel bytesPerRow:2];

    // Fallback pre-filtered environment map: 1x1 cube, neutral grey
    MTLTextureDescriptor* prefiltDesc =
        [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                             size:1
                                                        mipmapped:NO];
    prefiltDesc.usage = MTLTextureUsageShaderRead;
    m_fallbackPrefilteredMap = [m_device newTextureWithDescriptor:prefiltDesc];
    const uint32_t prefiltPixel = 0xff404040u;
    for (NSUInteger face = 0; face < 6; ++face)
    {
        [m_fallbackPrefilteredMap replaceRegion:region
                                    mipmapLevel:0
                                          slice:face
                                      withBytes:&prefiltPixel
                                    bytesPerRow:4
                                  bytesPerImage:4];
    }
}

void MetalRenderer::LoadIrradianceMap()
{
    const std::string irradiancePath = ResolveAssetPath("IrradianceMap_BC6U.dds");
    if (irradiancePath.empty())
    {
        NSLog(@"Irradiance map not found. Using fallback cubemap.");
        m_irradianceMap = m_fallbackIrradianceMap;
        return;
    }

    id<MTLTexture> irradiance = LoadCubeTextureOrNil(irradiancePath, false);
    m_irradianceMap = irradiance ? irradiance : m_fallbackIrradianceMap;
}

void MetalRenderer::LoadBrdfLut()
{
    const std::string brdfPath = ResolveAssetPath("BrdfLut.png");
    if (brdfPath.empty())
    {
        NSLog(@"BRDF LUT not found. Using fallback.");
        m_brdfLut = m_fallbackBrdfLut;
        return;
    }
    id<MTLTexture> lut = LoadTextureOrNil(brdfPath, false);
    m_brdfLut = lut ? lut : m_fallbackBrdfLut;
}

void MetalRenderer::LoadPrefilteredMap()
{
    const std::string prefiltPath = ResolveAssetPath("PrefilteredMap_BC6U.dds");
    if (prefiltPath.empty())
    {
        NSLog(@"Pre-filtered environment map not found. Using fallback cubemap.");
        m_prefilteredMap = m_fallbackPrefilteredMap;
        return;
    }
    id<MTLTexture> map = LoadCubeTextureOrNil(prefiltPath, false);
    m_prefilteredMap = map ? map : m_fallbackPrefilteredMap;
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

void MetalRenderer::DrawFrame()
{
    if (m_scene.IndexCount() == 0 || !m_vb || !m_ib) {
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

        const float dt = 0.014f;

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
        if (inp.KeyPressed(18)) // 1
        {
            m_enableVintagePostProcess = !m_enableVintagePostProcess;
            NSLog(@"Vintage post process %@", m_enableVintagePostProcess ? @"enabled" : @"disabled");
        }
        if (inp.KeyPressed(19)) // 2
        {
            m_enableChromaticAberration = !m_enableChromaticAberration;
            NSLog(@"Chromatic aberration %@", m_enableChromaticAberration ? @"enabled" : @"disabled");
        }
        if (inp.KeyPressed(20)) // 3
        {
            m_enableEyeAdaptationPostProcess = !m_enableEyeAdaptationPostProcess;
            NSLog(@"Eye Adaptation post process %@", m_enableEyeAdaptationPostProcess ? @"enabled" : @"disabled");
        }
        if (inp.KeyPressed(11)) // B
        {
            m_useBeckmannNDF = true;
            NSLog(@"NDF: Beckmann");
        }
        if (inp.KeyPressed(5)) // G
        {
            m_useBeckmannNDF = false;
            NSLog(@"NDF: GGX");
        }

        simd::float3 target = m_camPos + front;
        cb->view = LookAtRH(m_camPos, target, simd::float3{0, 1, 0});
        cb->cameraPos = m_camPos;
        m_timeSeconds += dt;

        const float enterPlaneDistance = m_model4PlaneSwapDistance;
        const float exitPlaneDistance = m_model4PlaneSwapDistance - m_model4PlaneSwapHysteresis;
        m_scene.UpdateModel4PlaneStates(m_camPos, enterPlaneDistance, exitPlaneDistance);

        cb->timeSeconds = 0.0f;

        // projection по размеру окна
        float w = (float)drawableWidth;
        float h = (float)drawableHeight;
        float aspect = (h > 0.0f) ? (w / h) : 1.0f;

        const float meshRadius = m_scene.MeshRadius();
        const float nearPlane = (meshRadius > 50.0f) ? 1.0f : 0.1f;
        const float terrainFarPlane = m_terrain.IsEnabled() ? (m_terrain.HalfSize() * 2.4f) : 0.0f;
        const float farPlane = fmaxf(nearPlane + 1.0f, fmaxf(meshRadius * 6.0f, terrainFarPlane));
        cb->proj = PerspectiveRH(60.0f * (float)M_PI / 180.0f, aspect, nearPlane, farPlane);

        cb->lightDir = m_directionalLight.GetDirection();
        cb->lightIntensity = m_directionalLight.GetIntensity();
        cb->lightColor = m_directionalLight.GetColor();
        cb->postProcessParams = simd::float4{
            m_enableVintagePostProcess ? 1.0f : 0.0f,
            m_timeSeconds,
            m_enableChromaticAberration ? 1.0f : 0.0f,
            0.0085f
        };
        cb->postProcessParams2 = simd::float4{
            m_enableEyeAdaptationPostProcess ? 1.0f : 0.0f,
            m_useBeckmannNDF ? 1.0f : 0.0f,
            0.0f,
            0.0f
        };

        const float cameraToMeshDistance = simd::distance(m_camPos, m_scene.MeshCenter());
        const float tessellationFadeNear =
            fmaxf(meshRadius * m_tessellationFadeNearMultiplier, 0.0f);
        const float tessellationFadeFar =
            fmaxf(tessellationFadeNear + 0.001f, meshRadius * m_tessellationFadeFarMultiplier);
        float tessellationFactor =
            1.0f - ((cameraToMeshDistance - tessellationFadeNear) /
                    (tessellationFadeFar - tessellationFadeNear));
        tessellationFactor = fmaxf(0.0f, fminf(tessellationFactor, 1.0f));
        const float tanHalfFovY = tanf(60.0f * (float)M_PI / 360.0f);
        const float tanHalfFovX = tanHalfFovY * aspect;
        m_terrain.UpdateMesh(cb->view,
                             nearPlane,
                             farPlane,
                             tanHalfFovX,
                             tanHalfFovY,
                             m_camPos,
                             m_enableFrustumCulling);

        id<MTLCommandBuffer> cmd = [m_queue commandBuffer];
        m_particleRenderer.Update(cmd, dt);

        m_scene.UpdateVisibility(cb->view,
                                 nearPlane,
                                 farPlane,
                                 tanHalfFovX,
                                 tanHalfFovY,
                                 m_enableFrustumCulling,
                                 m_enableBvhFrustumCulling);

        m_shadowRenderer.RenderCascades(cmd,
                                        m_dss,
                                        m_vb,
                                        m_sampler,
                                        m_camPos,
                                        front,
                                        m_directionalLight.GetDirection(),
                                        nearPlane,
                                        farPlane,
                                        tanHalfFovX,
                                        tanHalfFovY,
                                        [&](id<MTLRenderCommandEncoder> shadowEnc,
                                            const simd::float4x4& lightView,
                                            const simd::float4x4& lightProj,
                                            id<MTLRenderPipelineState> shadowPSO,
                                            id<MTLRenderPipelineState> fenceShadowPSO)
        {
            for (const Scene::DrawBatch& b : m_scene.Batches())
            {
                if (b.instanceIndex >= m_scene.VisibleInstances().size() || m_scene.VisibleInstances()[b.instanceIndex] == 0u)
                {
                    continue;
                }

                const Scene::Instance& instance = m_scene.Instances()[b.instanceIndex];
                if (instance.sourceModelIndex == 3u &&
                    b.instanceIndex < m_scene.Model4PlaneStates().size() &&
                    m_scene.Model4PlaneStates()[b.instanceIndex] != 0u)
                {
                    continue;
                }

                MaterialGPU mat{};
                if (b.materialIndex < m_scene.Materials().size())
                {
                    mat = m_scene.Materials()[b.materialIndex];
                }
                const float modelTessellationStrength = m_scene.GetTessellationStrengthForModel(b.sourceModelIndex);
                mat.detailParams.x = meshRadius * modelTessellationStrength * tessellationFactor;

                CameraCB shadowLocalCb = *cb;
                shadowLocalCb.view = lightView;
                shadowLocalCb.proj = lightProj;
                shadowLocalCb.world = simd_mul(MatTranslation(instance.worldOffset), MatScale(instance.scale));
                [shadowEnc setVertexBytes:&shadowLocalCb length:sizeof(CameraCB) atIndex:1];
                [shadowEnc setVertexBytes:&mat length:sizeof(MaterialGPU) atIndex:2];

                id<MTLTexture> heightTex = m_blackTex;
                if (b.materialIndex < m_scene.HeightTextures().size() && m_scene.HeightTextures()[b.materialIndex])
                {
                    heightTex = m_scene.HeightTextures()[b.materialIndex];
                }
                [shadowEnc setVertexTexture:heightTex atIndex:0];

                [shadowEnc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                      indexCount:b.indexCount
                                       indexType:MTLIndexTypeUInt32
                                     indexBuffer:m_ib
                               indexBufferOffset:(NSUInteger)b.indexOffset * sizeof(uint32_t)];
            }

            if (m_terrain.IsDrawable())
            {
                CameraCB terrainShadowCb = *cb;
                terrainShadowCb.view = lightView;
                terrainShadowCb.proj = lightProj;
                terrainShadowCb.world = matrix_identity_float4x4;

                [shadowEnc setVertexBuffer:m_terrain.VertexBuffer() offset:0 atIndex:0];
                [shadowEnc setVertexBytes:&terrainShadowCb length:sizeof(CameraCB) atIndex:1];
                const MaterialGPU& terrainMaterial = m_terrain.Material();
                [shadowEnc setVertexBytes:&terrainMaterial length:sizeof(MaterialGPU) atIndex:2];
                [shadowEnc setVertexTexture:m_blackTex atIndex:0];
                [shadowEnc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                      indexCount:m_terrain.IndexCount()
                                       indexType:MTLIndexTypeUInt32
                                     indexBuffer:m_terrain.IndexBuffer()
                               indexBufferOffset:0];

                [shadowEnc setVertexBuffer:m_vb offset:0 atIndex:0];
            }

            m_models.DrawFenceShadowCaster(shadowEnc,
                                           *cb,
                                           lightView,
                                           lightProj,
                                           fenceShadowPSO,
                                           shadowPSO,
                                           m_sampler,
                                           m_whiteTex,
                                           m_vb);
        });

        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:gbufferPass];


        [enc setRenderPipelineState:m_gbufferPSO];
        [enc setDepthStencilState:m_dss];

        [enc setVertexBuffer:m_vb offset:0 atIndex:0];
        [enc setFragmentBuffer:m_cameraCB offset:0 atIndex:0];
        [enc setFragmentSamplerState:m_sampler atIndex:0];
        [enc setVertexSamplerState:m_sampler atIndex:0];

        [enc setFragmentBuffer:m_appendStructuredBuffer offset:0 atIndex:3];
        [enc setFragmentBuffer:m_appendCounterBuffer offset:0 atIndex:4];
        [enc setFragmentBuffer:m_consumeStructuredBuffer offset:0 atIndex:5];
        [enc setFragmentBuffer:m_consumeCounterBuffer offset:0 atIndex:6];

        if (m_terrain.IsDrawable())
        {
            CameraCB terrainCb = *cb;
            terrainCb.world = matrix_identity_float4x4;
            const MaterialGPU& terrainMaterial = m_terrain.Material();

            [enc setVertexBuffer:m_terrain.VertexBuffer() offset:0 atIndex:0];
            [enc setVertexBytes:&terrainCb length:sizeof(CameraCB) atIndex:1];
            [enc setVertexBytes:&terrainMaterial length:sizeof(MaterialGPU) atIndex:2];
            [enc setFragmentBytes:&terrainMaterial length:sizeof(MaterialGPU) atIndex:1];
            [enc setVertexTexture:m_blackTex atIndex:0];

            for (const Terrain::DrawBatch& batch : m_terrain.Batches())
            {
                id<MTLTexture> diffuseTexture = m_whiteTex;
                id<MTLTexture> normalTexture = m_flatNormalTex;
                const Terrain::SourceTile* sourceTile = m_terrain.SourceTileAt(batch.sourceTileIndex);
                if (sourceTile && sourceTile->diffuseTexture)
                {
                    diffuseTexture = sourceTile->diffuseTexture;
                }
                if (sourceTile && sourceTile->normalTexture)
                {
                    normalTexture = sourceTile->normalTexture;
                }
                [enc setFragmentTexture:diffuseTexture atIndex:0];
                [enc setFragmentTexture:normalTexture atIndex:1];
                [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                indexCount:batch.indexCount
                                 indexType:MTLIndexTypeUInt32
                               indexBuffer:m_terrain.IndexBuffer()
                         indexBufferOffset:(NSUInteger)batch.indexOffset * sizeof(uint32_t)];
            }

            [enc setVertexBuffer:m_vb offset:0 atIndex:0];
        }

        for (const Scene::DrawBatch& b : m_scene.Batches())
        {
            if (b.instanceIndex >= m_scene.VisibleInstances().size() || m_scene.VisibleInstances()[b.instanceIndex] == 0u)
            {
                continue;
            }

            const Scene::Instance& instance = m_scene.Instances()[b.instanceIndex];
            const bool usePlaneForInstance =
                instance.sourceModelIndex == 3u &&
                b.instanceIndex < m_scene.Model4PlaneStates().size() &&
                m_scene.Model4PlaneStates()[b.instanceIndex] != 0u;
            if (usePlaneForInstance)
            {
                continue;
            }
            MaterialGPU mat{};
            if (b.materialIndex < m_scene.Materials().size())
            {
                mat = m_scene.Materials()[b.materialIndex];
            }
            const float modelTessellationStrength = m_scene.GetTessellationStrengthForModel(b.sourceModelIndex);
            const float displacementStrength =
                meshRadius * modelTessellationStrength * tessellationFactor;
            mat.detailParams.x = displacementStrength;

            CameraCB localCb = *cb;
            localCb.world = simd_mul(MatTranslation(instance.worldOffset), MatScale(instance.scale));
            [enc setVertexBytes:&localCb length:sizeof(CameraCB) atIndex:1];
            [enc setVertexBytes:&mat length:sizeof(MaterialGPU) atIndex:2];
            [enc setFragmentBytes:&mat length:sizeof(MaterialGPU) atIndex:1];

            id<MTLTexture> diffuseTex = m_whiteTex;
            if (b.materialIndex < m_scene.DiffuseTextures().size() && m_scene.DiffuseTextures()[b.materialIndex])
            {
                diffuseTex = m_scene.DiffuseTextures()[b.materialIndex];
            }
            id<MTLTexture> normalTex = m_flatNormalTex;
            if (b.materialIndex < m_scene.NormalTextures().size() && m_scene.NormalTextures()[b.materialIndex])
            {
                normalTex = m_scene.NormalTextures()[b.materialIndex];
            }
            id<MTLTexture> heightTex = m_blackTex;
            if (b.materialIndex < m_scene.HeightTextures().size() && m_scene.HeightTextures()[b.materialIndex])
            {
                heightTex = m_scene.HeightTextures()[b.materialIndex];
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

        m_models.DrawModel4Planes(enc,
                                  *cb,
                                  m_scene,
                                  m_camPos,
                                  m_blackTex,
                                  m_whiteTex,
                                  m_flatNormalTex,
                                  m_vb);

        m_particleRenderer.Draw(enc,
                                cb,
                                sizeof(CameraCB),
                                m_blackTex,
                                m_whiteTex,
                                m_flatNormalTex,
                                m_vb);

        m_models.DrawFence(enc, *cb, m_sampler, m_whiteTex);
        m_models.DrawFenceShadowDecal(enc, *cb, m_sampler);

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
        [lightingEnc setFragmentBuffer:m_shadowRenderer.ConstantBuffer() offset:0 atIndex:1];
        [lightingEnc setFragmentSamplerState:m_sampler atIndex:0];
        [lightingEnc setFragmentSamplerState:m_shadowRenderer.Sampler() atIndex:1];
        [lightingEnc setFragmentTexture:m_gbuffer.GetAlbedoTexture() atIndex:0];
        [lightingEnc setFragmentTexture:m_gbuffer.GetNormalTexture() atIndex:1];
        [lightingEnc setFragmentTexture:m_gbuffer.GetPositionTexture() atIndex:2];
        [lightingEnc setFragmentTexture:m_gbuffer.GetMaterialTexture() atIndex:3];
        [lightingEnc setFragmentTexture:m_shadowRenderer.Map(0) atIndex:4];
        [lightingEnc setFragmentTexture:m_shadowRenderer.Map(1) atIndex:5];
        [lightingEnc setFragmentTexture:m_shadowRenderer.Map(2) atIndex:6];
        [lightingEnc setFragmentTexture:m_shadowRenderer.Map(3) atIndex:7];
        [lightingEnc setFragmentTexture:(m_irradianceMap ? m_irradianceMap : m_fallbackIrradianceMap) atIndex:8];
        [lightingEnc setFragmentTexture:(m_brdfLut ? m_brdfLut : m_fallbackBrdfLut) atIndex:9];
        [lightingEnc setFragmentTexture:(m_prefilteredMap ? m_prefilteredMap : m_fallbackPrefilteredMap) atIndex:10];
        [lightingEnc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        [lightingEnc endEncoding];

        [cmd presentDrawable:drawable];
        [cmd commit];
    }
}
