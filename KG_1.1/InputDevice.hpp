#pragma once
#include <cstdint>
#include <array>

class InputDevice
{
public:
    static InputDevice& Get();

    void BeginFrame();
    void EndFrame();

    void OnKeyDown(uint16_t keyCode);
    void OnKeyUp(uint16_t keyCode);

    void OnMouseDown(int button);
    void OnMouseUp(int button);
    void OnMouseMove(float x, float y);

    bool KeyHeld(uint16_t keyCode) const;
    bool KeyPressed(uint16_t keyCode) const;
    bool KeyReleased(uint16_t keyCode) const;  

    float MouseX() const { return m_mouseX; }
    float MouseY() const { return m_mouseY; }
    float MouseDeltaX() const { return m_mouseX - m_prevMouseX; }
    float MouseDeltaY() const { return m_mouseY - m_prevMouseY; }

    bool MouseButtonHeld(int button) const;

    void SetModifierFlags(uint64_t flags) { m_modifierFlags = flags; }
    bool ModifierShift() const { return (m_modifierFlags & (1ULL << 17)) != 0; }
    bool ModifierControl() const { return (m_modifierFlags & (1ULL << 18)) != 0; }

private:
    std::array<bool, 256> m_curr{};
    std::array<bool, 256> m_prev{};

    float m_mouseX = 0.0f;
    float m_mouseY = 0.0f;
    float m_prevMouseX = 0.0f;
    float m_prevMouseY = 0.0f;

    static constexpr int kMaxMouseButtons = 8;
    std::array<bool, kMaxMouseButtons> m_mouseButtons{};

    uint64_t m_modifierFlags = 0;

    InputDevice() = default;
    
};
