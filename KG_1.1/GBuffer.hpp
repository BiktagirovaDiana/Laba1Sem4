#pragma once

#import <Metal/Metal.h>

class GBuffer
{
public:
    GBuffer(id<MTLDevice> device = nil);

    void SetDevice(id<MTLDevice> device);
    void EnsureTextures(NSUInteger width, NSUInteger height);
    bool IsReady() const;

    MTLRenderPassDescriptor* CreateRenderPassDescriptor() const;

    id<MTLTexture> GetAlbedoTexture() const;
    id<MTLTexture> GetNormalTexture() const;
    id<MTLTexture> GetPositionTexture() const;
    id<MTLTexture> GetMaterialTexture() const;
    id<MTLTexture> GetDepthTexture() const;

private:
    id<MTLDevice> m_device = nil;
    id<MTLTexture> m_albedoTexture = nil;
    id<MTLTexture> m_normalTexture = nil;
    id<MTLTexture> m_positionTexture = nil;
    id<MTLTexture> m_materialTexture = nil;
    id<MTLTexture> m_depthTexture = nil;
    NSUInteger m_width = 0;
    NSUInteger m_height = 0;
};
