//
//  InputDevice.cpp
//  KG_1.1
//
//  Created by Macbook on 21.02.2026.
//

#include "InputDevice.hpp"

InputDevice& InputDevice::Get()
{
    static InputDevice inst;
    return inst;
}

void InputDevice::BeginFrame()
{
    m_prev = m_curr;
}

void InputDevice::EndFrame()
{
    m_prevMouseX = m_mouseX;
    m_prevMouseY = m_mouseY;
}

void InputDevice::OnKeyDown(uint16_t keyCode)
{
    if (keyCode < m_curr.size()) m_curr[keyCode] = true;
}

void InputDevice::OnKeyUp(uint16_t keyCode)
{
    if (keyCode < m_curr.size()) m_curr[keyCode] = false;
}

void InputDevice::OnMouseDown(int button)
{
    if (button >= 0 && button < kMaxMouseButtons) m_mouseButtons[button] = true;
}

void InputDevice::OnMouseUp(int button)
{
    if (button >= 0 && button < kMaxMouseButtons) m_mouseButtons[button] = false;
}

void InputDevice::OnMouseMove(float x, float y)
{
    m_mouseX = x; m_mouseY = y;
}

bool InputDevice::KeyHeld(uint16_t keyCode) const
{
    return keyCode < m_curr.size() ? m_curr[keyCode] : false;
}

bool InputDevice::KeyPressed(uint16_t keyCode) const
{
    return keyCode < m_curr.size() ? (m_curr[keyCode] && !m_prev[keyCode]) : false;
}

bool InputDevice::KeyReleased(uint16_t keyCode) const
{
    return keyCode < m_curr.size() ? (!m_curr[keyCode] && m_prev[keyCode]) : false;
}

bool InputDevice::MouseButtonHeld(int button) const
{
    return (button >= 0 && button < kMaxMouseButtons) ? m_mouseButtons[button] : false;
}
