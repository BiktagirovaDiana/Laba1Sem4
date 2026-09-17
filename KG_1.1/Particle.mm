#import "Particle.hpp"
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstring>

static float SmoothStep01(float t)
{
    t = fmaxf(0.0f, fminf(t, 1.0f));
    return t * t * (3.0f - 2.0f * t);
}

static ObjMesh CreateUnitBillboardQuadMesh()
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
    return mesh;
}

static std::vector<ParticleRenderer::ParticleInstanceGPU> CreateParticleInstances(const ObjMesh& mesh)
{
    std::vector<ParticleRenderer::ParticleInstanceGPU> instances;
    instances.reserve(mesh.vertices.size() / 4u);

    for (size_t i = 0; i + 3 < mesh.vertices.size(); i += 4)
    {
        const VertexPNT& v0 = mesh.vertices[i + 0];
        const VertexPNT& v1 = mesh.vertices[i + 1];
        const VertexPNT& v2 = mesh.vertices[i + 2];
        const VertexPNT& v3 = mesh.vertices[i + 3];
        const simd::float3 center =
            (simd::float3{v0.px, v0.py, v0.pz} +
             simd::float3{v1.px, v1.py, v1.pz} +
             simd::float3{v2.px, v2.py, v2.pz} +
             simd::float3{v3.px, v3.py, v3.pz}) * 0.25f;
        const float width = simd::distance(simd::float3{v0.px, v0.py, v0.pz},
                                           simd::float3{v1.px, v1.py, v1.pz});
        const float height = simd::distance(simd::float3{v1.px, v1.py, v1.pz},
                                            simd::float3{v2.px, v2.py, v2.pz});

        ParticleRenderer::ParticleInstanceGPU instance;
        instance.baseCenterAndSize = simd::float4{center.x, center.y, center.z, fmaxf(fmaxf(width, height), 0.001f)};
        instance.animatedCenterAndSeed = simd::float4{center.x, center.y, center.z, (float)instances.size()};
        instances.push_back(instance);
    }

    return instances;
}

void ParticleRenderer::CreatePipelines(id<MTLDevice> device,
                                       id<MTLLibrary> library,
                                       MTLRenderPipelineDescriptor* gbufferPipelineDescriptor)
{
    NSError* err = nil;

    id<MTLFunction> vsParticleBillboard = [library newFunctionWithName:@"vs_particle_billboard"];
    MTLRenderPipelineDescriptor* particleDesc = [gbufferPipelineDescriptor copy];
    particleDesc.vertexFunction = vsParticleBillboard;
    m_billboardPSO = [device newRenderPipelineStateWithDescriptor:particleDesc error:&err];
    if (!m_billboardPSO) { NSLog(@"Particle billboard PSO error: %@", err); }

    id<MTLFunction> particleCompute = [library newFunctionWithName:@"cs_update_particles"];
    m_particleComputePSO = [device newComputePipelineStateWithFunction:particleCompute error:&err];
    if (!m_particleComputePSO) { NSLog(@"Particle compute PSO error: %@", err); }

    id<MTLFunction> dustCompute = [library newFunctionWithName:@"cs_update_dust_particles"];
    m_dustComputePSO = [device newComputePipelineStateWithFunction:dustCompute error:&err];
    if (!m_dustComputePSO) { NSLog(@"Dust particle compute PSO error: %@", err); }

    id<MTLFunction> rainCompute = [library newFunctionWithName:@"cs_update_rain_particles"];
    m_rainComputePSO = [device newComputePipelineStateWithFunction:rainCompute error:&err];
    if (!m_rainComputePSO) { NSLog(@"Rain particle compute PSO error: %@", err); }
}

void ParticleRenderer::CreateResources(id<MTLDevice> device,
                                       AssetResolver assetResolver,
                                       TextureLoader textureLoader)
{
    const std::string particleTexturePath = assetResolver ? assetResolver("Particle1.png") : std::string();
    const std::string dustTexturePath = assetResolver ? assetResolver("Particle2.png") : std::string();
    const std::string rainTexturePath = assetResolver ? assetResolver("Particle3.png") : std::string();

    ObjMesh quadMesh = CreateUnitBillboardQuadMesh();
    m_quadIndexCount = (uint32_t)quadMesh.indices.size();
    if (!quadMesh.vertices.empty() && !m_quadVB)
    {
        m_quadVB = [device newBufferWithBytes:quadMesh.vertices.data()
                                       length:quadMesh.vertices.size() * sizeof(VertexPNT)
                                      options:MTLResourceStorageModeShared];
    }
    if (!quadMesh.indices.empty() && !m_quadIB)
    {
        m_quadIB = [device newBufferWithBytes:quadMesh.indices.data()
                                       length:quadMesh.indices.size() * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];
    }

    CreateParticleResources(device, particleTexturePath, textureLoader);
    CreateDustParticleResources(device, dustTexturePath, textureLoader);
    CreateRainParticleResources(device, rainTexturePath, textureLoader);
}

void ParticleRenderer::CreateParticleResources(id<MTLDevice> device,
                                               const std::string& particleTexturePath,
                                               TextureLoader textureLoader)
{
    Particle particles(m_particlePlaneCount,
                       m_particleRadius,
                       m_particlePlaneSize,
                       m_particleCenter);
    ObjMesh particleMesh = particles.CreateMesh();

    std::vector<ParticleInstanceGPU> instances = CreateParticleInstances(particleMesh);
    m_particleInstanceCount = (uint32_t)instances.size();
    if (!instances.empty())
    {
        m_particleBaseInstanceBuffer = [device newBufferWithBytes:instances.data()
                                                           length:instances.size() * sizeof(ParticleInstanceGPU)
                                                          options:MTLResourceStorageModeShared];
        m_particleInstanceBuffer = [device newBufferWithBytes:instances.data()
                                                       length:instances.size() * sizeof(ParticleInstanceGPU)
                                                      options:MTLResourceStorageModeShared];
    }

    m_particleMaterial = {};
    m_particleMaterial.kd_ns = simd::float4{0.9f, 0.95f, 1.0f, 8.0f};
    m_particleMaterial.ks_alpha = simd::float4{0.0f, 0.0f, 0.0f, 1.0f};
    m_particleMaterial.uvScale = simd::float2{1.0f, 1.0f};
    m_particleMaterial.uvSpeed = simd::float2{0.0f, 0.0f};
    m_particleMaterial.textureFlags = simd::uint4{particleTexturePath.empty() ? 0u : 1u, 0u, 0u, 1u};
    m_particleMaterial.detailParams = simd::float4{0.0f, 1.0f, 0.0f, 0.0f};
    m_particleTexture = (particleTexturePath.empty() || !textureLoader) ? nil : textureLoader(particleTexturePath, true);
}

void ParticleRenderer::CreateDustParticleResources(id<MTLDevice> device,
                                                   const std::string& dustTexturePath,
                                                   TextureLoader textureLoader)
{
    Particle dustParticles(m_dustParticlePlaneCount,
                           m_dustParticleRadius,
                           m_dustParticlePlaneSize,
                           m_dustParticleCenter,
                           Particle::VolumeShape::Cube);
    ObjMesh dustMesh = dustParticles.CreateMesh();

    std::vector<ParticleInstanceGPU> instances = CreateParticleInstances(dustMesh);
    m_dustParticleInstanceCount = (uint32_t)instances.size();
    if (!instances.empty())
    {
        m_dustBaseInstanceBuffer = [device newBufferWithBytes:instances.data()
                                                       length:instances.size() * sizeof(ParticleInstanceGPU)
                                                      options:MTLResourceStorageModeShared];
        m_dustInstanceBuffer = [device newBufferWithBytes:instances.data()
                                                   length:instances.size() * sizeof(ParticleInstanceGPU)
                                                  options:MTLResourceStorageModeShared];
    }

    m_dustParticleMaterial = {};
    m_dustParticleMaterial.kd_ns = simd::float4{0.55f, 0.58f, 0.62f, 4.0f};
    m_dustParticleMaterial.ks_alpha = simd::float4{0.0f, 0.0f, 0.0f, 1.0f};
    m_dustParticleMaterial.uvScale = simd::float2{1.0f, 1.0f};
    m_dustParticleMaterial.uvSpeed = simd::float2{0.0f, 0.0f};
    m_dustParticleMaterial.textureFlags = simd::uint4{dustTexturePath.empty() ? 0u : 1u, 0u, 0u, 1u};
    m_dustParticleMaterial.detailParams = simd::float4{0.0f, 1.0f, 0.0f, 0.0f};
    m_dustParticleTexture = (dustTexturePath.empty() || !textureLoader) ? nil : textureLoader(dustTexturePath, true);
}

void ParticleRenderer::CreateRainParticleResources(id<MTLDevice> device,
                                                   const std::string& rainTexturePath,
                                                   TextureLoader textureLoader)
{
    Particle rainParticles(m_rainParticlePlaneCount,
                           m_rainParticleRadius,
                           m_rainParticlePlaneSize,
                           m_rainParticleCenter,
                           Particle::VolumeShape::Cube);
    ObjMesh rainMesh = rainParticles.CreateMesh();

    std::vector<ParticleInstanceGPU> instances = CreateParticleInstances(rainMesh);
    m_rainParticleInstanceCount = (uint32_t)instances.size();
    if (!instances.empty())
    {
        m_rainBaseInstanceBuffer = [device newBufferWithBytes:instances.data()
                                                       length:instances.size() * sizeof(ParticleInstanceGPU)
                                                      options:MTLResourceStorageModeShared];
        m_rainInstanceBuffer = [device newBufferWithBytes:instances.data()
                                                   length:instances.size() * sizeof(ParticleInstanceGPU)
                                                  options:MTLResourceStorageModeShared];
    }

    m_rainParticleMaterial = {};
    m_rainParticleMaterial.kd_ns = simd::float4{0.72f, 0.82f, 0.96f, 6.0f};
    m_rainParticleMaterial.ks_alpha = simd::float4{0.0f, 0.0f, 0.0f, 1.0f};
    m_rainParticleMaterial.uvScale = simd::float2{1.0f, 1.0f};
    m_rainParticleMaterial.uvSpeed = simd::float2{0.0f, 0.0f};
    m_rainParticleMaterial.textureFlags = simd::uint4{rainTexturePath.empty() ? 0u : 1u, 0u, 0u, 1u};
    m_rainParticleMaterial.detailParams = simd::float4{0.0f, 1.0f, 0.0f, 0.0f};
    m_rainParticleTexture = (rainTexturePath.empty() || !textureLoader) ? nil : textureLoader(rainTexturePath, true);
}

void ParticleRenderer::BuildRainCollisionPlanes(id<MTLDevice> device,
                                                const std::vector<CollisionAabb>& collisionAabbs)
{
    std::vector<RainCollisionPlaneGPU> planes;
    planes.reserve(collisionAabbs.size());

    for (const CollisionAabb& aabb : collisionAabbs)
    {
        const float minX = aabb.min.x;
        const float minZ = aabb.min.z;
        const float maxX = aabb.max.x;
        const float maxZ = aabb.max.z;
        const float topY = aabb.max.y;

        if ((maxX - minX) < 0.05f || (maxZ - minZ) < 0.05f)
        {
            continue;
        }

        RainCollisionPlaneGPU plane;
        plane.minXZMaxXZ = simd::float4{minX, minZ, maxX, maxZ};
        plane.yAndPadding = simd::float4{topY, 0.0f, 0.0f, 0.0f};
        planes.push_back(plane);
    }

    m_rainCollisionPlaneCount = (uint32_t)planes.size();
    m_rainCollisionPlaneBuffer = nil;
    if (!planes.empty())
    {
        m_rainCollisionPlaneBuffer = [device newBufferWithBytes:planes.data()
                                                         length:planes.size() * sizeof(RainCollisionPlaneGPU)
                                                        options:MTLResourceStorageModeShared];
    }
}

void ParticleRenderer::Update(id<MTLCommandBuffer> commandBuffer, float dt)
{
    UpdateParticleAnimation(commandBuffer, dt);
    UpdateDustParticleAnimation(commandBuffer, dt);
    UpdateRainParticleAnimation(commandBuffer, dt);
}

void ParticleRenderer::UpdateParticleAnimation(id<MTLCommandBuffer> commandBuffer, float dt)
{
    if (!commandBuffer || !m_particleComputePSO ||
        !m_particleBaseInstanceBuffer || !m_particleInstanceBuffer || m_particleInstanceCount == 0u)
    {
        return;
    }

    m_particleAnimationTime += dt;
    const float cycleSeconds = fmaxf(m_particleCycleSeconds, 0.001f);
    const float phase = fmodf(m_particleAnimationTime, cycleSeconds) / cycleSeconds;

    float sphereFactor = 1.0f;
    if (phase < m_particleCollapsePart)
    {
        sphereFactor = 1.0f - SmoothStep01(phase / fmaxf(m_particleCollapsePart, 0.001f));
    }
    else
    {
        const float explodePhase =
            (phase - m_particleCollapsePart) / fmaxf(1.0f - m_particleCollapsePart, 0.001f);
        sphereFactor = 1.0f - powf(1.0f - fmaxf(0.0f, fminf(explodePhase, 1.0f)), 5.0f);
    }

    ParticleAnimationCB particleCb;
    particleCb.centerAndFactor =
        simd::float4{m_particleCenter.x, m_particleCenter.y, m_particleCenter.z, sphereFactor};
    particleCb.instanceCount = m_particleInstanceCount;

    id<MTLComputeCommandEncoder> computeEnc = [commandBuffer computeCommandEncoder];
    if (!computeEnc)
    {
        return;
    }

    [computeEnc setComputePipelineState:m_particleComputePSO];
    [computeEnc setBuffer:m_particleBaseInstanceBuffer offset:0 atIndex:0];
    [computeEnc setBuffer:m_particleInstanceBuffer offset:0 atIndex:1];
    [computeEnc setBytes:&particleCb length:sizeof(ParticleAnimationCB) atIndex:2];

    const NSUInteger threadCount = (NSUInteger)m_particleInstanceCount;
    const NSUInteger maxThreadsPerGroup = m_particleComputePSO.maxTotalThreadsPerThreadgroup;
    const NSUInteger threadsPerGroup = std::min<NSUInteger>(maxThreadsPerGroup, 256u);
    [computeEnc dispatchThreads:MTLSizeMake(threadCount, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
    [computeEnc endEncoding];
}

void ParticleRenderer::UpdateDustParticleAnimation(id<MTLCommandBuffer> commandBuffer, float dt)
{
    if (!commandBuffer || !m_dustComputePSO ||
        !m_dustBaseInstanceBuffer || !m_dustInstanceBuffer || m_dustParticleInstanceCount == 0u)
    {
        return;
    }

    m_dustAnimationTime += dt;

    DustAnimationCB dustCb;
    dustCb.centerAndTime =
        simd::float4{m_dustParticleCenter.x, m_dustParticleCenter.y, m_dustParticleCenter.z, m_dustAnimationTime};
    dustCb.motionParams =
        simd::float4{m_dustDriftAmplitude, m_dustDriftSpeed, m_dustSwirlAmplitude, 0.0f};
    dustCb.instanceCount = m_dustParticleInstanceCount;

    id<MTLComputeCommandEncoder> computeEnc = [commandBuffer computeCommandEncoder];
    if (!computeEnc)
    {
        return;
    }

    [computeEnc setComputePipelineState:m_dustComputePSO];
    [computeEnc setBuffer:m_dustBaseInstanceBuffer offset:0 atIndex:0];
    [computeEnc setBuffer:m_dustInstanceBuffer offset:0 atIndex:1];
    [computeEnc setBytes:&dustCb length:sizeof(DustAnimationCB) atIndex:2];

    const NSUInteger threadCount = (NSUInteger)m_dustParticleInstanceCount;
    const NSUInteger maxThreadsPerGroup = m_dustComputePSO.maxTotalThreadsPerThreadgroup;
    const NSUInteger threadsPerGroup = std::min<NSUInteger>(maxThreadsPerGroup, 256u);
    [computeEnc dispatchThreads:MTLSizeMake(threadCount, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
    [computeEnc endEncoding];
}

void ParticleRenderer::UpdateRainParticleAnimation(id<MTLCommandBuffer> commandBuffer, float dt)
{
    if (!commandBuffer || !m_rainComputePSO ||
        !m_rainBaseInstanceBuffer || !m_rainInstanceBuffer || m_rainParticleInstanceCount == 0u)
    {
        return;
    }

    m_rainAnimationTime += dt;

    RainAnimationCB rainCb;
    rainCb.centerAndTime =
        simd::float4{m_rainParticleCenter.x, m_rainParticleCenter.y, m_rainParticleCenter.z, m_rainAnimationTime};
    rainCb.volumeAndSpeed =
        simd::float4{m_rainParticleRadius, m_rainFallHeight, m_rainFallSpeed, 0.0f};
    rainCb.bounceParams =
        simd::float4{m_rainBounceHeight, m_rainBounceDistance, m_rainCollisionBias, 0.0f};
    rainCb.instanceCount = m_rainParticleInstanceCount;
    rainCb.collisionPlaneCount = m_rainCollisionPlaneCount;

    id<MTLComputeCommandEncoder> computeEnc = [commandBuffer computeCommandEncoder];
    if (!computeEnc)
    {
        return;
    }

    [computeEnc setComputePipelineState:m_rainComputePSO];
    [computeEnc setBuffer:m_rainBaseInstanceBuffer offset:0 atIndex:0];
    [computeEnc setBuffer:m_rainInstanceBuffer offset:0 atIndex:1];
    [computeEnc setBytes:&rainCb length:sizeof(RainAnimationCB) atIndex:2];
    if (m_rainCollisionPlaneBuffer && m_rainCollisionPlaneCount > 0u)
    {
        [computeEnc setBuffer:m_rainCollisionPlaneBuffer offset:0 atIndex:3];
    }
    else
    {
        [computeEnc setBuffer:nil offset:0 atIndex:3];
    }

    const NSUInteger threadCount = (NSUInteger)m_rainParticleInstanceCount;
    const NSUInteger maxThreadsPerGroup = m_rainComputePSO.maxTotalThreadsPerThreadgroup;
    const NSUInteger threadsPerGroup = std::min<NSUInteger>(maxThreadsPerGroup, 256u);
    [computeEnc dispatchThreads:MTLSizeMake(threadCount, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
    [computeEnc endEncoding];
}

void ParticleRenderer::Draw(id<MTLRenderCommandEncoder> encoder,
                            const void* cameraConstantBuffer,
                            NSUInteger cameraConstantBufferLength,
                            id<MTLTexture> blackTexture,
                            id<MTLTexture> whiteTexture,
                            id<MTLTexture> flatNormalTexture,
                            id<MTLBuffer> defaultVertexBuffer)
{
    DrawParticleSet(encoder,
                    cameraConstantBuffer,
                    cameraConstantBufferLength,
                    m_particleMaterial,
                    m_particleInstanceBuffer,
                    m_particleTexture,
                    m_particleInstanceCount,
                    blackTexture,
                    whiteTexture,
                    flatNormalTexture);
    DrawParticleSet(encoder,
                    cameraConstantBuffer,
                    cameraConstantBufferLength,
                    m_dustParticleMaterial,
                    m_dustInstanceBuffer,
                    m_dustParticleTexture,
                    m_dustParticleInstanceCount,
                    blackTexture,
                    whiteTexture,
                    flatNormalTexture);
    DrawParticleSet(encoder,
                    cameraConstantBuffer,
                    cameraConstantBufferLength,
                    m_rainParticleMaterial,
                    m_rainInstanceBuffer,
                    m_rainParticleTexture,
                    m_rainParticleInstanceCount,
                    blackTexture,
                    whiteTexture,
                    flatNormalTexture);

    [encoder setVertexBuffer:defaultVertexBuffer offset:0 atIndex:0];
}

void ParticleRenderer::DrawParticleSet(id<MTLRenderCommandEncoder> encoder,
                                       const void* cameraConstantBuffer,
                                       NSUInteger cameraConstantBufferLength,
                                       const MaterialGPU& material,
                                       id<MTLBuffer> instanceBuffer,
                                       id<MTLTexture> texture,
                                       uint32_t instanceCount,
                                       id<MTLTexture> blackTexture,
                                       id<MTLTexture> whiteTexture,
                                       id<MTLTexture> flatNormalTexture)
{
    if (!m_billboardPSO || !m_quadVB || !m_quadIB ||
        !instanceBuffer || m_quadIndexCount == 0u || instanceCount == 0u ||
        !cameraConstantBuffer || cameraConstantBufferLength < sizeof(simd::float4x4))
    {
        return;
    }

    std::vector<uint8_t> localCamera(cameraConstantBufferLength);
    std::memcpy(localCamera.data(), cameraConstantBuffer, cameraConstantBufferLength);
    std::memcpy(localCamera.data(), &matrix_identity_float4x4, sizeof(simd::float4x4));

    [encoder setRenderPipelineState:m_billboardPSO];
    [encoder setVertexBuffer:m_quadVB offset:0 atIndex:0];
    [encoder setVertexBytes:localCamera.data() length:cameraConstantBufferLength atIndex:1];
    [encoder setVertexBytes:&material length:sizeof(MaterialGPU) atIndex:2];
    [encoder setVertexBuffer:instanceBuffer offset:0 atIndex:3];
    [encoder setFragmentBytes:&material length:sizeof(MaterialGPU) atIndex:1];
    [encoder setVertexTexture:blackTexture atIndex:0];
    [encoder setFragmentTexture:(texture ? texture : whiteTexture) atIndex:0];
    [encoder setFragmentTexture:flatNormalTexture atIndex:1];

    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:m_quadIndexCount
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:m_quadIB
                 indexBufferOffset:0
                     instanceCount:instanceCount];
}
