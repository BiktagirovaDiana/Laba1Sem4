#pragma once
#include <string>

class MetalRenderer;

class Window
{
public:
    Window(int width, int height, const std::string& title);
    ~Window();

    void Show();
    bool Pump();
    void RequestClose();

    void* GetNativeWindow() const;
    int   Width()  const { return m_width; }
    int   Height() const { return m_height; }

private:
    int  m_width = 0;
    int  m_height = 0;
    bool m_shouldClose = false;
    
    

    void* m_nsWindow = nullptr;
    void* m_mtkView  = nullptr;
    void* m_delegate = nullptr;
    
    void* m_viewDelegate = nullptr;

    MetalRenderer* m_renderer = nullptr;

    void CreateCocoaWindow(const std::string& title);
};
