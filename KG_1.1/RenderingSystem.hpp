#pragma once
#import <MetalKit/MetalKit.h>

class MetalRenderer;

class RenderingSystem
{
public:
    explicit RenderingSystem(MTKView* view);
    ~RenderingSystem();

    void DrawFrame();

private:
    MetalRenderer* m_renderer = nullptr;
};
