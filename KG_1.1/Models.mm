#import "Models.hpp"
#include "MatrixFactory.hpp"
#import <Foundation/Foundation.h>

#include <cmath>
#include <vector>

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

void Models::CreatePipelines(id<MTLDevice> device,
                             id<MTLLibrary> library,
                             MTLVertexDescriptor* vertexDescriptor,
                             MTLPixelFormat depthPixelFormat)
{
    NSError* err = nil;
    id<MTLFunction> vertexShaderFenceGbuf = [library newFunctionWithName:@"vertex_shader_fence_gbuffer"];
    id<MTLFunction> fragmentShaderFenceGbuf = [library newFunctionWithName:@"fragment_shader_fence_gbuffer"];
    MTLRenderPipelineDescriptor* fenceGbufDesc = [MTLRenderPipelineDescriptor new];
    fenceGbufDesc.vertexFunction = vertexShaderFenceGbuf;
    fenceGbufDesc.fragmentFunction = fragmentShaderFenceGbuf;
    fenceGbufDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    fenceGbufDesc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA16Float;
    fenceGbufDesc.colorAttachments[2].pixelFormat = MTLPixelFormatRGBA16Float;
    fenceGbufDesc.colorAttachments[3].pixelFormat = MTLPixelFormatRGBA16Float;
    fenceGbufDesc.depthAttachmentPixelFormat = depthPixelFormat;
    fenceGbufDesc.vertexDescriptor = vertexDescriptor;

    m_fenceGbufferPSO = [device newRenderPipelineStateWithDescriptor:fenceGbufDesc error:&err];
    if (!m_fenceGbufferPSO) { NSLog(@"Fence GBuffer PSO error: %@", err); }
}

void Models::CreateResources(id<MTLDevice> device,
                             AssetResolver assetResolver,
                             TextureLoader textureLoader)
{
    const std::string model4TexturePath = assetResolver ? assetResolver("model4.png") : std::string();
    ObjMesh planeMesh = CreateTexturedPlaneMesh(model4TexturePath);
    m_model4PlaneIndexCount = (uint32_t)planeMesh.indices.size();
    if (!planeMesh.vertices.empty())
    {
        m_model4PlaneVB = [device newBufferWithBytes:planeMesh.vertices.data()
                                              length:planeMesh.vertices.size() * sizeof(VertexPNT)
                                             options:MTLResourceStorageModeShared];
    }
    if (!planeMesh.indices.empty())
    {
        m_model4PlaneIB = [device newBufferWithBytes:planeMesh.indices.data()
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
    m_model4PlaneTexture = (!model4TexturePath.empty() && textureLoader) ? textureLoader(model4TexturePath, true) : nil;

    const std::string fenceTexturePath = assetResolver ? assetResolver("realistic-steel-fence.png") : std::string();
    if (fenceTexturePath.empty())
    {
        NSLog(@"[Fence] Texture 'realistic-steel-fence.png' not found in assets.");
    }
    m_fenceTexture = (!fenceTexturePath.empty() && textureLoader) ? textureLoader(fenceTexturePath, true) : nil;

    const std::vector<VertexPNT> verts =
    {
        { -0.5f, -0.5f, 0.0f,  0.0f, 0.0f, 1.0f,  0.0f, 1.0f },
        {  0.5f, -0.5f, 0.0f,  0.0f, 0.0f, 1.0f,  1.0f, 1.0f },
        {  0.5f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f,  1.0f, 0.0f },
        { -0.5f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f },
    };
    const std::vector<uint32_t> indices = {0u, 1u, 2u, 0u, 2u, 3u};
    m_fenceIndexCount = (uint32_t)indices.size();
    m_fenceVB = [device newBufferWithBytes:verts.data()
                                    length:verts.size() * sizeof(VertexPNT)
                                   options:MTLResourceStorageModeShared];
    m_fenceIB = [device newBufferWithBytes:indices.data()
                                    length:indices.size() * sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];

    m_fenceMaterial = {};
    m_fenceMaterial.kd_ns = simd::float4{1.0f, 1.0f, 1.0f, 32.0f};
    m_fenceMaterial.ks_alpha = simd::float4{0.05f, 0.05f, 0.05f, 1.0f};
    m_fenceMaterial.uvScale = simd::float2{1.0f, 1.0f};
    m_fenceMaterial.uvSpeed = simd::float2{0.0f, 0.0f};
    m_fenceMaterial.textureFlags = simd::uint4{m_fenceTexture ? 1u : 0u, 0u, 0u, 0u};
    m_fenceMaterial.detailParams = simd::float4{0.0f, 1.0f, 0.0f, 0.0f};

    m_fenceShadowMaterial = {};
    m_fenceShadowMaterial.kd_ns = simd::float4{0.0f, 0.0f, 0.0f, 1.0f};
    m_fenceShadowMaterial.ks_alpha = simd::float4{0.0f, 0.0f, 0.0f, 1.0f};
    m_fenceShadowMaterial.uvScale = simd::float2{1.0f, 1.0f};
    m_fenceShadowMaterial.uvSpeed = simd::float2{0.0f, 0.0f};
    m_fenceShadowMaterial.textureFlags = simd::uint4{m_fenceTexture ? 1u : 0u, 0u, 0u, 0u};
    m_fenceShadowMaterial.detailParams = simd::float4{0.0f, 1.0f, 0.0f, 0.0f};
}

void Models::DrawFenceShadowCaster(id<MTLRenderCommandEncoder> encoder,
                                   const CameraCB& camera,
                                   const simd::float4x4& lightViewMatrix,
                                   const simd::float4x4& lightProjectionMatrix,
                                   id<MTLRenderPipelineState> fenceShadowPSO,
                                   id<MTLRenderPipelineState> restorePSO,
                                   id<MTLSamplerState> sampler,
                                   id<MTLTexture> whiteTexture,
                                   id<MTLBuffer> restoreVertexBuffer) const
{
    if (!fenceShadowPSO || !m_fenceVB || !m_fenceIB || m_fenceIndexCount == 0u)
    {
        return;
    }

    CameraCB fenceShadowCb = camera;
    fenceShadowCb.viewMatrix = lightViewMatrix;
    fenceShadowCb.projectionMatrix = lightProjectionMatrix;
    fenceShadowCb.modelMatrix = MatrixFactory::FenceWorld(m_fencePosition, m_fenceSize,
                                                    m_fenceYawRadians,
                                                    m_fencePitchRadians,
                                                    m_fenceRollRadians);

    MaterialGPU fenceShadowMat = m_fenceMaterial;
    fenceShadowMat.detailParams.x = 0.0f;

    [encoder setRenderPipelineState:fenceShadowPSO];
    [encoder setCullMode:MTLCullModeNone];
    [encoder setVertexBuffer:m_fenceVB offset:0 atIndex:0];
    [encoder setVertexBytes:&fenceShadowCb length:sizeof(CameraCB) atIndex:1];
    [encoder setVertexBytes:&fenceShadowMat length:sizeof(MaterialGPU) atIndex:2];
    [encoder setFragmentBytes:&fenceShadowMat length:sizeof(MaterialGPU) atIndex:2];
    [encoder setFragmentTexture:(m_fenceTexture ? m_fenceTexture : whiteTexture) atIndex:0];
    [encoder setFragmentSamplerState:sampler atIndex:0];

    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:m_fenceIndexCount
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:m_fenceIB
                 indexBufferOffset:0];

    if (restorePSO)
    {
        [encoder setRenderPipelineState:restorePSO];
    }
    [encoder setCullMode:MTLCullModeBack];
    [encoder setVertexBuffer:restoreVertexBuffer offset:0 atIndex:0];
}

void Models::DrawModel4Planes(id<MTLRenderCommandEncoder> encoder,
                              const CameraCB& camera,
                              const Scene& scene,
                              simd::float3 cameraPosition,
                              id<MTLTexture> blackTexture,
                              id<MTLTexture> whiteTexture,
                              id<MTLTexture> flatNormalTexture,
                              id<MTLBuffer> restoreVertexBuffer) const
{
    if (!m_model4PlaneVB || !m_model4PlaneIB || m_model4PlaneIndexCount == 0u)
    {
        return;
    }

    [encoder setVertexBuffer:m_model4PlaneVB offset:0 atIndex:0];
    [encoder setVertexTexture:blackTexture atIndex:0];
    [encoder setFragmentTexture:(m_model4PlaneTexture ? m_model4PlaneTexture : whiteTexture) atIndex:0];
    [encoder setFragmentTexture:flatNormalTexture atIndex:1];

    for (uint32_t instanceIndex = 0; instanceIndex < scene.Instances().size(); ++instanceIndex)
    {
        if (instanceIndex >= scene.VisibleInstances().size() || scene.VisibleInstances()[instanceIndex] == 0u)
        {
            continue;
        }
        if (instanceIndex >= scene.Model4PlaneStates().size() || scene.Model4PlaneStates()[instanceIndex] == 0u)
        {
            continue;
        }

        const Scene::Instance& instance = scene.Instances()[instanceIndex];
        if (instance.sourceModelIndex != 3u)
        {
            continue;
        }

        CameraCB localCb = camera;
        localCb.modelMatrix = simd_mul(simd_mul(MatrixFactory::Translation(instance.worldOffset),
                                          MatrixFactory::BillboardFacingCamera(instance.worldOffset,
                                                                               cameraPosition)),
                                 MatrixFactory::UniformScale(m_model4PlaneScale));
        [encoder setVertexBytes:&localCb length:sizeof(CameraCB) atIndex:1];
        [encoder setVertexBytes:&m_model4PlaneMaterial length:sizeof(MaterialGPU) atIndex:2];
        [encoder setFragmentBytes:&m_model4PlaneMaterial length:sizeof(MaterialGPU) atIndex:1];

        [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:m_model4PlaneIndexCount
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:m_model4PlaneIB
                     indexBufferOffset:0];
    }

    [encoder setVertexBuffer:restoreVertexBuffer offset:0 atIndex:0];
}

void Models::DrawFence(id<MTLRenderCommandEncoder> encoder,
                       const CameraCB& camera,
                       id<MTLSamplerState> sampler,
                       id<MTLTexture> whiteTexture) const
{
    if (!m_fenceGbufferPSO || !m_fenceVB || !m_fenceIB || m_fenceIndexCount == 0u)
    {
        return;
    }

    CameraCB fenceCb = camera;
    fenceCb.modelMatrix = MatrixFactory::FenceWorld(m_fencePosition, m_fenceSize,
                                              m_fenceYawRadians,
                                              m_fencePitchRadians,
                                              m_fenceRollRadians);

    [encoder setRenderPipelineState:m_fenceGbufferPSO];
    [encoder setCullMode:MTLCullModeNone];
    [encoder setVertexBuffer:m_fenceVB offset:0 atIndex:0];
    [encoder setVertexBytes:&fenceCb length:sizeof(CameraCB) atIndex:1];
    [encoder setVertexBytes:&m_fenceMaterial length:sizeof(MaterialGPU) atIndex:2];
    [encoder setFragmentBytes:&fenceCb length:sizeof(CameraCB) atIndex:0];
    [encoder setFragmentBytes:&m_fenceMaterial length:sizeof(MaterialGPU) atIndex:1];
    [encoder setFragmentTexture:(m_fenceTexture ? m_fenceTexture : whiteTexture) atIndex:0];
    [encoder setFragmentSamplerState:sampler atIndex:0];

    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:m_fenceIndexCount
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:m_fenceIB
                 indexBufferOffset:0];

    [encoder setCullMode:MTLCullModeBack];
}

void Models::DrawFenceShadowDecal(id<MTLRenderCommandEncoder> encoder,
                                  const CameraCB& camera,
                                  id<MTLSamplerState> sampler) const
{
    if (!m_fenceGbufferPSO || !m_fenceVB || !m_fenceIB || m_fenceIndexCount == 0u || !m_fenceTexture)
    {
        return;
    }

    CameraCB shadowCb = camera;
    shadowCb.modelMatrix = MatrixFactory::FenceWorld(m_fenceShadowPosition, m_fenceShadowSize,
                                               m_fenceShadowYawRadians,
                                               m_fenceShadowPitchRadians,
                                               m_fenceShadowRollRadians);

    [encoder setRenderPipelineState:m_fenceGbufferPSO];
    [encoder setCullMode:MTLCullModeNone];
    [encoder setVertexBuffer:m_fenceVB offset:0 atIndex:0];
    [encoder setVertexBytes:&shadowCb length:sizeof(CameraCB) atIndex:1];
    [encoder setVertexBytes:&m_fenceShadowMaterial length:sizeof(MaterialGPU) atIndex:2];
    [encoder setFragmentBytes:&shadowCb length:sizeof(CameraCB) atIndex:0];
    [encoder setFragmentBytes:&m_fenceShadowMaterial length:sizeof(MaterialGPU) atIndex:1];
    [encoder setFragmentTexture:m_fenceTexture atIndex:0];
    [encoder setFragmentSamplerState:sampler atIndex:0];

    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:m_fenceIndexCount
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:m_fenceIB
                 indexBufferOffset:0];

    [encoder setCullMode:MTLCullModeBack];
}
