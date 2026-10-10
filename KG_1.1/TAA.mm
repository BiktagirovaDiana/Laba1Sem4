#import "TAA.hpp"

TAA::TAA() = default;

void TAA::CreatePipelines(id<MTLDevice> device, id<MTLLibrary> library, MTLPixelFormat pixelFormat)
{
    if (!device || !library)
    {
        return;
    }

    NSError* error = nil;
    id<MTLFunction> vertexShaderFullscreen = [library newFunctionWithName:@"vertex_shader_fullscreen"];
    id<MTLFunction> fragmentShaderTaaResolve = [library newFunctionWithName:@"fragment_shader_taa_resolve"];
    id<MTLFunction> fragmentShaderCopyTexture = [library newFunctionWithName:@"fragment_shader_copy_texture"];

    MTLRenderPipelineDescriptor* resolveDesc = [MTLRenderPipelineDescriptor new];
    resolveDesc.vertexFunction = vertexShaderFullscreen;
    resolveDesc.fragmentFunction = fragmentShaderTaaResolve;
    resolveDesc.colorAttachments[0].pixelFormat = pixelFormat;
    m_resolvePSO = [device newRenderPipelineStateWithDescriptor:resolveDesc error:&error];
    if (!m_resolvePSO)
    {
        NSLog(@"TAA resolve PSO error: %@", error);
    }

    error = nil;
    MTLRenderPipelineDescriptor* copyDesc = [MTLRenderPipelineDescriptor new];
    copyDesc.vertexFunction = vertexShaderFullscreen;
    copyDesc.fragmentFunction = fragmentShaderCopyTexture;
    copyDesc.colorAttachments[0].pixelFormat = pixelFormat;
    m_copyPSO = [device newRenderPipelineStateWithDescriptor:copyDesc error:&error];
    if (!m_copyPSO)
    {
        NSLog(@"TAA copy PSO error: %@", error);
    }
}
//проверка
void TAA::EnsureResources(id<MTLDevice> device,
                          NSUInteger width,
                          NSUInteger height,
                          MTLPixelFormat pixelFormat)
{
    if (!device || width == 0 || height == 0)
    {
        return;
    }

    if (m_currentColorTexture &&
        m_width == width &&
        m_height == height &&
        m_pixelFormat == pixelFormat)
    {
        return;
    }

    m_width = width;
    m_height = height;
    m_pixelFormat = pixelFormat;
    m_historyIndex = 0;
    m_hasHistory = false;

    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

    m_currentColorTexture = [device newTextureWithDescriptor:desc];
    m_historyTextures[0] = [device newTextureWithDescriptor:desc];
    m_historyTextures[1] = [device newTextureWithDescriptor:desc];
}

void TAA::ResetHistory()
{
    m_hasHistory = false;
}

bool TAA::IsReady() const
{
    return m_resolvePSO &&
           m_copyPSO &&
           m_currentColorTexture &&
           m_historyTextures[0] &&
           m_historyTextures[1];
}

id<MTLTexture> TAA::CurrentColorTexture() const
{
    return m_currentColorTexture;
}

MTLRenderPassDescriptor* TAA::CreateCurrentColorPassDescriptor() const
{
    if (!m_currentColorTexture)
    {
        return nil;
    }

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = m_currentColorTexture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.5, 0.7, 1.0, 1.0);
    return pass;
}

void TAA::EncodeResolve(id<MTLCommandBuffer> commandBuffer,
                        MTLRenderPassDescriptor* drawablePass,
                        id<MTLSamplerState> sampler,
                        id<MTLTexture> albedoTexture,
                        id<MTLTexture> velocityTexture,
                        const ResolveParams& params)
{
    if (!IsReady() || !commandBuffer || !drawablePass || !sampler ||
        !albedoTexture || !velocityTexture)
    {
        return;
    }

    const uint32_t readIndex = m_historyIndex;
    const uint32_t writeIndex = 1u - m_historyIndex;

    MTLRenderPassDescriptor* historyPass = [MTLRenderPassDescriptor renderPassDescriptor];
    historyPass.colorAttachments[0].texture = m_historyTextures[writeIndex];
    historyPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    historyPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    historyPass.colorAttachments[0].clearColor = MTLClearColorMake(0.5, 0.7, 1.0, 1.0);

    ResolveParams localParams = params;
    if (m_hasHistory)
    {
        localParams.flags.x = params.flags.x;
    }
    else
    {
        localParams.flags.x = 1.0f;
    }

    id<MTLRenderCommandEncoder> resolveEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:historyPass];
    [resolveEncoder setRenderPipelineState:m_resolvePSO];
    [resolveEncoder setFragmentSamplerState:sampler atIndex:0];
    [resolveEncoder setFragmentBytes:&localParams length:sizeof(ResolveParams) atIndex:0];
    [resolveEncoder setFragmentTexture:m_currentColorTexture atIndex:0];
    [resolveEncoder setFragmentTexture:m_historyTextures[readIndex] atIndex:1];
    [resolveEncoder setFragmentTexture:albedoTexture atIndex:2];
    [resolveEncoder setFragmentTexture:velocityTexture atIndex:3];
    [resolveEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [resolveEncoder endEncoding];

    id<MTLRenderCommandEncoder> copyEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:drawablePass];
    [copyEncoder setRenderPipelineState:m_copyPSO];
    [copyEncoder setFragmentSamplerState:sampler atIndex:0];
    [copyEncoder setFragmentTexture:m_historyTextures[writeIndex] atIndex:0];
    [copyEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [copyEncoder endEncoding];

    m_historyIndex = writeIndex;
    m_hasHistory = true;
}
