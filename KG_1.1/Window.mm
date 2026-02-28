//
//  Window.mm
//  KG_1.1
//
//  Created by Macbook on 21.02.2026.
//

#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>
#include <CoreFoundation/CoreFoundation.h>

#include "Window.hpp"
#include "MetalRenderer.hpp"
#include "InputDevice.hpp"

static inline NSWindow* AsNSWindow(void* p) { return (__bridge NSWindow*)p; }
static inline MTKView*  AsMTKView(void* p)  { return (__bridge MTKView*)p; }

static inline void* RetainToVoid(id obj) { return (__bridge_retained void*)obj; }

static inline void ReleaseVoid(void*& p)
{
    if (p)
    {
        CFRelease(p);
        p = nullptr;
    }
}



@interface WindowDelegate : NSObject <NSWindowDelegate>
{
@public
    Window* owner;
}
@end

@implementation WindowDelegate
- (void)windowWillClose:(NSNotification*)notification
{
    owner->RequestClose();
}
@end

@interface GameView : MTKView
@end

@implementation GameView
- (BOOL)acceptsFirstResponder { return YES; }

- (void)keyDown:(NSEvent*)event
{
    InputDevice::Get().SetModifierFlags((uint64_t)event.modifierFlags);
    InputDevice::Get().OnKeyDown((uint16_t)event.keyCode);
}
- (void)keyUp:(NSEvent*)event
{
    InputDevice::Get().SetModifierFlags((uint64_t)event.modifierFlags);
    InputDevice::Get().OnKeyUp((uint16_t)event.keyCode);
}

- (void)mouseDown:(NSEvent*)event
{
    InputDevice::Get().SetModifierFlags((uint64_t)event.modifierFlags);
    InputDevice::Get().OnMouseDown((int)event.buttonNumber);
}
- (void)mouseUp:(NSEvent*)event
{
    InputDevice::Get().SetModifierFlags((uint64_t)event.modifierFlags);
    InputDevice::Get().OnMouseUp((int)event.buttonNumber);
}

- (void)mouseMoved:(NSEvent*)event
{
    InputDevice::Get().SetModifierFlags((uint64_t)event.modifierFlags);
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    InputDevice::Get().OnMouseMove((float)p.x, (float)p.y);
}

- (void)mouseDragged:(NSEvent*)event
{
    InputDevice::Get().SetModifierFlags((uint64_t)event.modifierFlags);
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    InputDevice::Get().OnMouseMove((float)p.x, (float)p.y);
}

- (void)rightMouseDown:(NSEvent*)event
{
    InputDevice::Get().SetModifierFlags((uint64_t)event.modifierFlags);
    InputDevice::Get().OnMouseDown((int)event.buttonNumber);
}

- (void)rightMouseUp:(NSEvent*)event
{
    InputDevice::Get().SetModifierFlags((uint64_t)event.modifierFlags);
    InputDevice::Get().OnMouseUp((int)event.buttonNumber);
}

- (void)rightMouseDragged:(NSEvent*)event
{
    InputDevice::Get().SetModifierFlags((uint64_t)event.modifierFlags);
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    InputDevice::Get().OnMouseMove((float)p.x, (float)p.y);
}
@end

@interface RendererDelegate : NSObject<MTKViewDelegate>
{
@public
    MetalRenderer* renderer;
}
@end

@implementation RendererDelegate
- (void)drawInMTKView:(MTKView *)view
{
    if (renderer) renderer->DrawFrame();
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size
{
    (void)view; (void)size;
}
@end



Window::Window(int width, int height, const std::string& title)
    : m_width(width), m_height(height)
{
    CreateCocoaWindow(title);

    m_renderer = new MetalRenderer(AsMTKView(m_mtkView));
    
    RendererDelegate* rd = (RendererDelegate*)[AsMTKView(m_mtkView) delegate];
    if (rd)
    {
        rd->renderer = m_renderer;
    }
}

Window::~Window()
{
    delete m_renderer;
    m_renderer = nullptr;

    if (m_nsWindow)
        [AsNSWindow(m_nsWindow) close];

    ReleaseVoid(m_viewDelegate);
    ReleaseVoid(m_delegate);
    ReleaseVoid(m_mtkView);
    ReleaseVoid(m_nsWindow);
}

void Window::CreateCocoaWindow(const std::string& title)
{
    NSString* nsTitle = [NSString stringWithUTF8String:title.c_str()];
    NSRect rect = NSMakeRect(100, 100, m_width, m_height);

    NSWindowStyleMask style =
        NSWindowStyleMaskTitled |
        NSWindowStyleMaskClosable |
        NSWindowStyleMaskResizable;

    NSWindow* win = [[NSWindow alloc] initWithContentRect:rect
                                                styleMask:style
                                                  backing:NSBackingStoreBuffered
                                                    defer:NO];
    [win setTitle:nsTitle];
    [win makeKeyAndOrderFront:nil];

    WindowDelegate* del = [WindowDelegate new];
    del->owner = this;
    [win setDelegate:del];

    GameView* view = [[GameView alloc] initWithFrame:rect];
    [win setContentView:view];
    [win makeFirstResponder:view];
    
    RendererDelegate* rd = [RendererDelegate new];
    rd->renderer = nullptr;
    view.delegate = rd;

    [win setAcceptsMouseMovedEvents:YES];

    m_nsWindow = RetainToVoid(win);
    m_mtkView  = RetainToVoid(view);
    m_delegate = RetainToVoid(del);
    m_viewDelegate = RetainToVoid(rd);
}

void Window::Show()
{
    if (m_nsWindow)
    {
        [AsNSWindow(m_nsWindow) makeKeyAndOrderFront:nil];
    }
}

bool Window::Pump()
{
    InputDevice::Get().BeginFrame();

    [AsMTKView(m_mtkView) draw];

    InputDevice::Get().EndFrame();
    return !m_shouldClose;
}

void Window::RequestClose()
{
    m_shouldClose = true;
}

void* Window::GetNativeWindow() const
{
    return m_nsWindow;
}
