//
//  Phong.metal
//  KG_1.1
//
//  Created by Macbook on 21.02.2026.
//

#include <metal_stdlib>
using namespace metal;

struct VertexIn
{
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

struct CameraCB
{
    float4x4 world;
    float4x4 view;
    float4x4 proj;
    float3   lightDir;
    float    pad0;
    float3   cameraPos;
    float    timeSeconds;
};

struct MaterialCB
{
    float4 kd_ns;
    float4 ks_alpha;
    float2 uvScale;
    float2 uvSpeed;
    uint useTexture;
    float3 pad;
};

struct VSOut
{
    float4 position [[position]];
    float3 worldPos;
    float3 worldN;
    float2 uv;
};

vertex VSOut vs_main(VertexIn vin [[stage_in]],
                     constant CameraCB& cb [[buffer(1)]])
{
    VSOut o;
    float4 wp = cb.world * float4(vin.position, 1.0);
    float4 vp = cb.view  * wp;
    o.position = cb.proj * vp;

    o.worldPos = wp.xyz;
    o.worldN = normalize((cb.world * float4(vin.normal, 0.0)).xyz);
    o.uv = vin.uv;
    return o;
}

fragment float4 ps_main(VSOut in [[stage_in]],
                        constant CameraCB& cb [[buffer(0)]],
                        constant MaterialCB& mat [[buffer(1)]],
                        texture2d<float> diffuseTex [[texture(0)]],
                        sampler linearSampler [[sampler(0)]])
{
    float3 N = normalize(in.worldN);
    float3 L = normalize(-cb.lightDir);
    float3 V = normalize(cb.cameraPos - in.worldPos);
    float3 R = reflect(-L, N);

    float  ambient = 0.10;
    float  diff = max(dot(N, L), 0.0);
    float  spec = pow(max(dot(R, V), 0.0), max(mat.kd_ns.w, 1.0));

    float2 uv = in.uv * mat.uvScale + mat.uvSpeed * cb.timeSeconds;
    
    float3 baseColor = mat.kd_ns.rgb;
    if (mat.useTexture != 0)
    {
        baseColor *= diffuseTex.sample(linearSampler, uv).rgb;
    }
    float3 color = baseColor * (ambient + diff) + mat.ks_alpha.rgb * spec;

    return float4(color, mat.ks_alpha.a);
}

//шейдер неба
struct SkyVSOut
{
    float4 position [[position]];
    float2 uv;
};

vertex SkyVSOut vs_sky(uint vid [[vertex_id]])
{
    SkyVSOut o;
    
    float2 pos[4] =
    {
        float2(-1.0, -1.0),  // нижний левый
        float2( 1.0, -1.0),   // нижний правый
        float2(-1.0,  1.0),   // верхний левый
        float2( 1.0,  1.0)    // верхний правый
    };
    float2 uv[4] = {
        float2(0.0, 1.0),     // нижний левый
        float2(1.0, 1.0),     // нижний правый
        float2(0.0, 0.0),     // верхний левый
        float2(1.0, 0.0)      // верхний правый
    };
    o.position = float4(pos[vid], 1.0, 1.0); // z=1.0 (дальняя плоскость NDC)
    o.uv = uv[vid];
    return o;
}

fragment float4 ps_sky(SkyVSOut in [[stage_in]]) //градиент неба
{
    float3 topColor = float3(0.5, 0.7, 1.0);
    float3 bottomColor = float3(0.7, 0.85, 1.0);
    
    float t = 1.0 - in.uv.y;
    float3 color = mix(bottomColor, topColor, t);
    
    return float4(color, 1.0);
}
