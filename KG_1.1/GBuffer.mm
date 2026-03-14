#import "GBuffer.hpp"

GBuffer::GBuffer(id<MTLDevice> device)
    : m_device(device)
{
}

void GBuffer::SetDevice(id<MTLDevice> device)
{
    m_device = device;
}

void GBuffer::EnsureTextures(NSUInteger width, NSUInteger height)
{
    if (!m_device || width == 0 || height == 0)
    {
        return;
    }

    if (m_albedoTexture && m_width == width && m_height == height)
    {
        return;
    }

    m_width = width;
    m_height = height;

    MTLTextureDescriptor* colorDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    colorDesc.storageMode = MTLStorageModePrivate;
    colorDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

    MTLTextureDescriptor* albedoDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    albedoDesc.storageMode = MTLStorageModePrivate;
    albedoDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

    MTLTextureDescriptor* depthDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    depthDesc.storageMode = MTLStorageModePrivate;
    depthDesc.usage = MTLTextureUsageRenderTarget;

    m_albedoTexture = [m_device newTextureWithDescriptor:albedoDesc];
    m_normalTexture = [m_device newTextureWithDescriptor:colorDesc];
    m_positionTexture = [m_device newTextureWithDescriptor:colorDesc];
    m_materialTexture = [m_device newTextureWithDescriptor:colorDesc];
    m_depthTexture = [m_device newTextureWithDescriptor:depthDesc];
}

bool GBuffer::IsReady() const
{
    return m_albedoTexture && m_normalTexture && m_positionTexture && m_materialTexture && m_depthTexture;
}

MTLRenderPassDescriptor* GBuffer::CreateRenderPassDescriptor() const
{
    if (!IsReady())
    {
        return nil;
    }

    MTLRenderPassDescriptor* renderPass = [MTLRenderPassDescriptor renderPassDescriptor];
    renderPass.colorAttachments[0].texture = m_albedoTexture;
    renderPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    renderPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    renderPass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

    renderPass.colorAttachments[1].texture = m_normalTexture;
    renderPass.colorAttachments[1].loadAction = MTLLoadActionClear;
    renderPass.colorAttachments[1].storeAction = MTLStoreActionStore;
    renderPass.colorAttachments[1].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

    renderPass.colorAttachments[2].texture = m_positionTexture;
    renderPass.colorAttachments[2].loadAction = MTLLoadActionClear;
    renderPass.colorAttachments[2].storeAction = MTLStoreActionStore;
    renderPass.colorAttachments[2].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

    renderPass.colorAttachments[3].texture = m_materialTexture;
    renderPass.colorAttachments[3].loadAction = MTLLoadActionClear;
    renderPass.colorAttachments[3].storeAction = MTLStoreActionStore;
    renderPass.colorAttachments[3].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

    renderPass.depthAttachment.texture = m_depthTexture;
    renderPass.depthAttachment.loadAction = MTLLoadActionClear;
    renderPass.depthAttachment.storeAction = MTLStoreActionDontCare;
    renderPass.depthAttachment.clearDepth = 1.0;

    return renderPass;
}

id<MTLTexture> GBuffer::GetAlbedoTexture() const
{
    return m_albedoTexture;
}

id<MTLTexture> GBuffer::GetNormalTexture() const
{
    return m_normalTexture;
}

id<MTLTexture> GBuffer::GetPositionTexture() const
{
    return m_positionTexture;
}

id<MTLTexture> GBuffer::GetMaterialTexture() const
{
    return m_materialTexture;
}

id<MTLTexture> GBuffer::GetDepthTexture() const
{
    return m_depthTexture;
}
