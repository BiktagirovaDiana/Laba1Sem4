//
//  main.mm
//  KG_1.1
//
//  Created by Macbook on 21.02.2026.
//

#import <Cocoa/Cocoa.h>
#include "Window.hpp"

int main(int argc, const char* argv[])
{

    @autoreleasepool
    {
        //для cocoa создание приложениея
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        Window window(1280, 720, "Diana's project");
        window.Show();

        //вывод приложение поверх других
        [NSApp activateIgnoringOtherApps:YES];

        bool running = true;
        
        while (running)
        {

            @autoreleasepool
            {
                NSEvent* event = nil;
                
                do {
                    event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                               untilDate:[NSDate dateWithTimeIntervalSinceNow:0.0]
                                                  inMode:NSDefaultRunLoopMode
                                                 dequeue:YES];
                    if (event)
                    {
                        //передача событие в Cocoa(клавиатура, мышь, окно).
                        [NSApp sendEvent:event];
                    }
                }
                while (event != nil);
                running = window.Pump();
            }
        }
    }
    return 0;
}
