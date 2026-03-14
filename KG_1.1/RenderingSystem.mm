#import "RenderingSystem.hpp"

#include "MetalRenderer.hpp"

RenderingSystem::RenderingSystem(MTKView* view)
{
    m_renderer = new MetalRenderer(view);
}

RenderingSystem::~RenderingSystem()
{
    delete m_renderer;
    m_renderer = nullptr;
}

void RenderingSystem::DrawFrame()
{
    if (m_renderer)
    {
        m_renderer->DrawFrame();
    }
}
