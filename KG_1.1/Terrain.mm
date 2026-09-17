#import "Terrain.hpp"
#import <Foundation/Foundation.h>

#include <sys/stat.h>
#include <algorithm>
#include <cmath>

static bool FileExists(const char* p)
{
    struct stat st;
    return (stat(p, &st) == 0) && S_ISREG(st.st_mode);
}

static std::string DirName(const std::string& p)
{
    const size_t slashPos = p.find_last_of("/\\");
    if (slashPos == std::string::npos)
    {
        return std::string();
    }
    return p.substr(0, slashPos);
}

static std::string JoinPath(const std::string& a, const std::string& b)
{
    if (a.empty())
    {
        return b;
    }
    if (a.back() == '/')
    {
        return a + b;
    }
    return a + "/" + b;
}

static std::vector<std::string> GetAssetCandidateDirs()
{
    std::vector<std::string> candidateDirs;

    const std::string sourceDir = DirName(__FILE__);
    if (!sourceDir.empty())
    {
        candidateDirs.push_back(JoinPath(sourceDir, "assets"));
    }

    candidateDirs.push_back("assets");
    candidateDirs.push_back("KG_1.1/assets");

    NSString* exePathNs = [[NSBundle mainBundle] executablePath];
    if (exePathNs)
    {
        const std::string exeDir = DirName([exePathNs UTF8String]);
        if (!exeDir.empty())
        {
            candidateDirs.push_back(JoinPath(exeDir, "assets"));
            candidateDirs.push_back(JoinPath(exeDir, "../assets"));
            candidateDirs.push_back(JoinPath(exeDir, "../Resources/assets"));
        }
    }

    NSString* resourcePathNs = [[NSBundle mainBundle] resourcePath];
    if (resourcePathNs)
    {
        const std::string resourceDir = [resourcePathNs UTF8String];
        candidateDirs.push_back(JoinPath(resourceDir, "assets"));
        candidateDirs.push_back(resourceDir);
    }

    return candidateDirs;
}

static std::string ResolveAssetPath(const std::string& fileName)
{
    for (const std::string& d : GetAssetCandidateDirs())
    {
        const std::string full = JoinPath(d, fileName);
        if (FileExists(full.c_str()))
        {
            return full;
        }
    }

    return std::string();
}

static std::string ResolveTerrainTileAssetPath(const std::string& fileName)
{
    std::string path = ResolveAssetPath(JoinPath("tiles", fileName));
    if (!path.empty())
    {
        return path;
    }
    return ResolveAssetPath(JoinPath("Tiles", fileName));
}

static float SmoothStep01(float t)
{
    t = fmaxf(0.0f, fminf(t, 1.0f));
    return t * t * (3.0f - 2.0f * t);
}

static bool IsSphereVisibleInFrustum(const simd::float4x4& viewMatrix,
                                     simd::float3 worldCenter,
                                     float radius,
                                     float nearPlane,
                                     float farPlane,
                                     float tanHalfFovX,
                                     float tanHalfFovY)
{
    const simd::float4 centerView4 = viewMatrix * simd::float4{worldCenter.x, worldCenter.y, worldCenter.z, 1.0f};
    const simd::float3 centerView = simd::float3{centerView4.x, centerView4.y, centerView4.z};
    const float depth = -centerView.z;

    if (depth + radius < nearPlane)
    {
        return false;
    }
    if (depth - radius > farPlane)
    {
        return false;
    }
    if (fabsf(centerView.x) > depth * tanHalfFovX + radius)
    {
        return false;
    }
    if (fabsf(centerView.y) > depth * tanHalfFovY + radius)
    {
        return false;
    }

    return true;
}

void Terrain::CreateResources(id<MTLDevice> device, TextureLoader textureLoader)
{
    m_device = device;
    m_textureLoader = std::move(textureLoader);
    LoadTiles();

    m_material = {};
    m_material.kd_ns = simd::float4{1.0f, 1.0f, 1.0f, 18.0f};
    m_material.ks_alpha = simd::float4{0.015f, 0.018f, 0.012f, 1.0f};
    m_material.uvScale = simd::float2{1.0f, 1.0f};
    m_material.uvSpeed = simd::float2{0.0f, 0.0f};
    m_material.textureFlags = simd::uint4{1u, 1u, 0u, 0u};
    m_material.detailParams = simd::float4{0.0f, 0.65f, 0.0f, 0.0f};
}

const Terrain::SourceTile* Terrain::SourceTileAt(uint32_t index) const
{
    if (index >= m_sourceTiles.size())
    {
        return nullptr;
    }
    return &m_sourceTiles[index];
}

Terrain::SourceTile Terrain::LoadSourceTile(const std::string& heightPath,
                                            const std::string& diffusePath)
{
    SourceTile tile;

    if (!heightPath.empty())
    {
        NSString* nsPath = [NSString stringWithUTF8String:heightPath.c_str()];
        NSData* imageData = [NSData dataWithContentsOfFile:nsPath];
        NSBitmapImageRep* bitmap = imageData ? [NSBitmapImageRep imageRepWithData:imageData] : nil;
        if (bitmap)
        {
            const NSInteger width = bitmap.pixelsWide;
            const NSInteger height = bitmap.pixelsHigh;
            const NSInteger bitsPerSample = std::max<NSInteger>(bitmap.bitsPerSample, 1);
            const float maxSampleValue = (bitsPerSample >= 16) ? 65535.0f : 255.0f;

            tile.width = (uint32_t)width;
            tile.height = (uint32_t)height;
            tile.heights.resize((size_t)width * (size_t)height);

            NSUInteger pixel[4] = {0, 0, 0, 0};
            for (NSInteger y = 0; y < height; ++y)
            {
                for (NSInteger x = 0; x < width; ++x)
                {
                    [bitmap getPixel:pixel atX:x y:y];
                    const float normalizedHeight =
                        fmaxf(0.0f, fminf((float)pixel[0] / maxSampleValue, 1.0f));
                    tile.heights[(size_t)y * (size_t)width + (size_t)x] = normalizedHeight;
                }
            }
        }
        else
        {
            NSLog(@"Terrain heightmap load failed: %@", nsPath);
        }
    }

    if (!diffusePath.empty() && m_textureLoader)
    {
        tile.diffuseTexture = m_textureLoader(diffusePath, true);
    }
    tile.normalTexture = CreateNormalTexture(tile);

    return tile;
}

id<MTLTexture> Terrain::CreateNormalTexture(const SourceTile& tile)
{
    if (tile.heights.empty() || tile.width == 0u || tile.height == 0u)
    {
        return nil;
    }

    std::vector<uint8_t> pixels((size_t)tile.width * (size_t)tile.height * 4u);
    const float normalStrength = 18.0f;
    auto sampleHeight = [&](uint32_t x, uint32_t y) -> float
    {
        x = std::min(x, tile.width - 1u);
        y = std::min(y, tile.height - 1u);
        return tile.heights[(size_t)y * tile.width + x];
    };

    for (uint32_t y = 0; y < tile.height; ++y)
    {
        for (uint32_t x = 0; x < tile.width; ++x)
        {
            const uint32_t xL = (x > 0u) ? x - 1u : x;
            const uint32_t xR = std::min(x + 1u, tile.width - 1u);
            const uint32_t yD = (y > 0u) ? y - 1u : y;
            const uint32_t yU = std::min(y + 1u, tile.height - 1u);
            const float hL = sampleHeight(xL, y);
            const float hR = sampleHeight(xR, y);
            const float hD = sampleHeight(x, yD);
            const float hU = sampleHeight(x, yU);

            const simd::float3 n =
                simd::normalize(simd::float3{(hL - hR) * normalStrength,
                                             (hD - hU) * normalStrength,
                                             1.0f});
            const size_t pixelOffset = ((size_t)y * tile.width + x) * 4u;
            pixels[pixelOffset + 0u] = (uint8_t)lrintf(fmaxf(0.0f, fminf(n.x * 0.5f + 0.5f, 1.0f)) * 255.0f);
            pixels[pixelOffset + 1u] = (uint8_t)lrintf(fmaxf(0.0f, fminf(n.y * 0.5f + 0.5f, 1.0f)) * 255.0f);
            pixels[pixelOffset + 2u] = (uint8_t)lrintf(fmaxf(0.0f, fminf(n.z * 0.5f + 0.5f, 1.0f)) * 255.0f);
            pixels[pixelOffset + 3u] = 255u;
        }
    }

    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:tile.width
                                                          height:tile.height
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [m_device newTextureWithDescriptor:desc];
    if (!texture)
    {
        return nil;
    }

    const MTLRegion region = MTLRegionMake2D(0, 0, tile.width, tile.height);
    [texture replaceRegion:region
               mipmapLevel:0
                 withBytes:pixels.data()
               bytesPerRow:(NSUInteger)tile.width * 4u];
    return texture;
}

void Terrain::LoadTiles()
{
    m_sourceTiles.clear();
    m_sourceTiles.reserve(9);

    for (uint32_t i = 1; i <= 9; ++i)
    {
        const std::string suffix = std::to_string(i) + ".png";
        const std::string heightPath = ResolveTerrainTileAssetPath("heightmap_16bit-" + suffix);
        const std::string diffusePath = ResolveTerrainTileAssetPath("satellite-" + suffix);

        SourceTile tile = LoadSourceTile(heightPath, diffusePath);
        if (tile.heights.empty())
        {
            NSLog(@"Terrain tile %u has no heightmap. height=%s diffuse=%s",
                  i,
                  heightPath.empty() ? "<missing>" : heightPath.c_str(),
                  diffusePath.empty() ? "<missing>" : diffusePath.c_str());
        }
        m_sourceTiles.push_back(tile);
    }
}

uint32_t Terrain::GetSourceTileIndex(float x, float z) const
{
    if (m_sourceTiles.empty())
    {
        return 0u;
    }

    const float terrainSize = m_halfSize * 2.0f;
    const float sourceTileSize = terrainSize / 3.0f;
    const int col = std::max(0, std::min(2, (int)floorf((x + m_halfSize) / sourceTileSize)));
    const int row = std::max(0, std::min(2, (int)floorf((z + m_halfSize) / sourceTileSize)));
    const uint32_t index = (uint32_t)(row * 3 + col);
    return std::min<uint32_t>(index, (uint32_t)m_sourceTiles.size() - 1u);
}

float Terrain::SampleHeightFromTile(const SourceTile& tile, float u, float v) const
{
    if (tile.heights.empty() || tile.width == 0u || tile.height == 0u)
    {
        return 0.5f;
    }

    u = fmaxf(0.0f, fminf(u, 1.0f));
    v = fmaxf(0.0f, fminf(v, 1.0f));

    const float fx = u * (float)(tile.width - 1u);
    const float fy = v * (float)(tile.height - 1u);
    const uint32_t x0 = (uint32_t)floorf(fx);
    const uint32_t y0 = (uint32_t)floorf(fy);
    const uint32_t x1 = std::min(x0 + 1u, tile.width - 1u);
    const uint32_t y1 = std::min(y0 + 1u, tile.height - 1u);
    const float tx = fx - (float)x0;
    const float ty = fy - (float)y0;

    const float h00 = tile.heights[(size_t)y0 * tile.width + x0];
    const float h10 = tile.heights[(size_t)y0 * tile.width + x1];
    const float h01 = tile.heights[(size_t)y1 * tile.width + x0];
    const float h11 = tile.heights[(size_t)y1 * tile.width + x1];
    const float hx0 = h00 + (h10 - h00) * tx;
    const float hx1 = h01 + (h11 - h01) * tx;
    return hx0 + (hx1 - hx0) * ty;
}

float Terrain::SampleHeight(float x, float z) const
{
    if (!m_sourceTiles.empty())
    {
        const float terrainSize = m_halfSize * 2.0f;
        const float sourceTileSize = terrainSize / 3.0f;
        const uint32_t sourceIndex = GetSourceTileIndex(x, z);
        const uint32_t col = sourceIndex % 3u;
        const uint32_t row = sourceIndex / 3u;
        const float sourceMinX = -m_halfSize + (float)col * sourceTileSize;
        const float sourceMinZ = -m_halfSize + (float)row * sourceTileSize;
        const float u = (x - sourceMinX) / sourceTileSize;
        const float v = (z - sourceMinZ) / sourceTileSize;
        const float height01 = SampleHeightFromTile(m_sourceTiles[sourceIndex], u, v);
        return m_baseY + (height01 - 0.5f) * (m_maxHeight * 2.0f);
    }

    const float broad = sinf(x * 0.018f) * cosf(z * 0.014f) * 7.0f;
    const float ridges = sinf((x + z) * 0.047f) * 2.4f;
    const float detail = (sinf(x * 0.11f + 1.7f) + cosf(z * 0.095f - 0.4f)) * 0.85f;
    const float centerFlatten = SmoothStep01(simd::length(simd::float2{x, z}) / 150.0f);
    const float height = fmaxf(-m_maxHeight, fminf(broad + ridges + detail, m_maxHeight));
    return m_baseY + height * centerFlatten;
}

simd::float3 Terrain::SampleNormal(float x, float z) const
{
    const float step = 1.0f;
    const float hL = SampleHeight(x - step, z);
    const float hR = SampleHeight(x + step, z);
    const float hD = SampleHeight(x, z - step);
    const float hU = SampleHeight(x, z + step);
    return simd::normalize(simd::float3{hL - hR, 2.0f * step, hD - hU});
}

void Terrain::SelectTilesRecursive(simd::float2 center,
                                   float size,
                                   uint32_t depth,
                                   const simd::float4x4& viewMatrix,
                                   float nearPlane,
                                   float farPlane,
                                   float tanHalfFovX,
                                   float tanHalfFovY,
                                   simd::float3 cameraPosition,
                                   bool enableFrustumCulling,
                                   std::vector<Tile>& outTiles) const
{
    if (enableFrustumCulling &&
        !IsTileVisibleInFrustum(center,
                                size,
                                viewMatrix,
                                nearPlane,
                                farPlane,
                                tanHalfFovX,
                                tanHalfFovY))
    {
        return;
    }

    const simd::float2 cameraXZ = simd::float2{cameraPosition.x, cameraPosition.z};
    const float distanceToTile = simd::length(cameraXZ - center);
    const bool shouldSplit =
        depth < m_maxDepth &&
        distanceToTile < size * m_lodDistanceFactor;

    if (!shouldSplit)
    {
        Tile tile;
        tile.center = center;
        tile.size = size;
        tile.depth = depth;
        outTiles.push_back(tile);
        return;
    }

    const float childSize = size * 0.5f;
    const float childOffset = size * 0.25f;
    SelectTilesRecursive(center + simd::float2{-childOffset, -childOffset},
                         childSize,
                         depth + 1u,
                         viewMatrix,
                         nearPlane,
                         farPlane,
                         tanHalfFovX,
                         tanHalfFovY,
                         cameraPosition,
                         enableFrustumCulling,
                         outTiles);
    SelectTilesRecursive(center + simd::float2{ childOffset, -childOffset},
                         childSize,
                         depth + 1u,
                         viewMatrix,
                         nearPlane,
                         farPlane,
                         tanHalfFovX,
                         tanHalfFovY,
                         cameraPosition,
                         enableFrustumCulling,
                         outTiles);
    SelectTilesRecursive(center + simd::float2{-childOffset,  childOffset},
                         childSize,
                         depth + 1u,
                         viewMatrix,
                         nearPlane,
                         farPlane,
                         tanHalfFovX,
                         tanHalfFovY,
                         cameraPosition,
                         enableFrustumCulling,
                         outTiles);
    SelectTilesRecursive(center + simd::float2{ childOffset,  childOffset},
                         childSize,
                         depth + 1u,
                         viewMatrix,
                         nearPlane,
                         farPlane,
                         tanHalfFovX,
                         tanHalfFovY,
                         cameraPosition,
                         enableFrustumCulling,
                         outTiles);
}

bool Terrain::IsTileVisibleInFrustum(simd::float2 center,
                                     float size,
                                     const simd::float4x4& viewMatrix,
                                     float nearPlane,
                                     float farPlane,
                                     float tanHalfFovX,
                                     float tanHalfFovY) const
{
    const simd::float3 tileCenter = simd::float3{center.x, m_baseY, center.y};
    const float horizontalRadius = size * 0.70710678f;
    const float verticalRadius = m_maxHeight;
    const float radius = sqrtf(horizontalRadius * horizontalRadius +
                               verticalRadius * verticalRadius);
    return IsSphereVisibleInFrustum(viewMatrix,
                                    tileCenter,
                                    radius,
                                    nearPlane,
                                    farPlane,
                                    tanHalfFovX,
                                    tanHalfFovY);
}

void Terrain::UpdateMesh(const simd::float4x4& viewMatrix,
                         float nearPlane,
                         float farPlane,
                         float tanHalfFovX,
                         float tanHalfFovY,
                         simd::float3 cameraPosition,
                         bool enableFrustumCulling)
{
    if (!m_enabled || m_patchResolution == 0u)
    {
        m_vertexBuffer = nil;
        m_indexBuffer = nil;
        m_indexCount = 0u;
        m_tileCount = 0u;
        m_batches.clear();
        return;
    }

    std::vector<Tile> tiles;
    tiles.reserve(256);
    const float terrainSize = m_halfSize * 2.0f;
    const float sourceTileSize = terrainSize / 3.0f;
    for (uint32_t row = 0; row < 3u; ++row)
    {
        for (uint32_t col = 0; col < 3u; ++col)
        {
            const simd::float2 sourceCenter =
                simd::float2{-m_halfSize + ((float)col + 0.5f) * sourceTileSize,
                             -m_halfSize + ((float)row + 0.5f) * sourceTileSize};
            SelectTilesRecursive(sourceCenter,
                                 sourceTileSize,
                                 0u,
                                 viewMatrix,
                                 nearPlane,
                                 farPlane,
                                 tanHalfFovX,
                                 tanHalfFovY,
                                 cameraPosition,
                                 enableFrustumCulling,
                                 tiles);
        }
    }

    std::vector<VertexPNT> vertices;
    std::vector<uint32_t> indices;
    const uint32_t r = m_patchResolution;
    const uint32_t vertsPerSide = r + 1u;
    vertices.reserve(tiles.size() * vertsPerSide * vertsPerSide);
    indices.reserve(tiles.size() * r * r * 6u);
    m_batches.clear();
    m_batches.reserve(tiles.size());

    for (const Tile& tile : tiles)
    {
        const uint32_t vertexBase = (uint32_t)vertices.size();
        const uint32_t indexBase = (uint32_t)indices.size();
        const float minX = tile.center.x - tile.size * 0.5f;
        const float minZ = tile.center.y - tile.size * 0.5f;
        const uint32_t sourceTileIndex = GetSourceTileIndex(tile.center.x, tile.center.y);
        const uint32_t sourceCol = sourceTileIndex % 3u;
        const uint32_t sourceRow = sourceTileIndex / 3u;
        const float sourceMinX = -m_halfSize + (float)sourceCol * sourceTileSize;
        const float sourceMinZ = -m_halfSize + (float)sourceRow * sourceTileSize;

        for (uint32_t z = 0; z <= r; ++z)
        {
            const float fz = (float)z / (float)r;
            const float worldZ = minZ + fz * tile.size;
            for (uint32_t x = 0; x <= r; ++x)
            {
                const float fx = (float)x / (float)r;
                const float worldX = minX + fx * tile.size;
                const float worldY = SampleHeight(worldX, worldZ);
                const simd::float3 n = SampleNormal(worldX, worldZ);
                const float u = (worldX - sourceMinX) / sourceTileSize;
                const float v = (worldZ - sourceMinZ) / sourceTileSize;
                vertices.push_back(VertexPNT{worldX,
                                             worldY,
                                             worldZ,
                                             n.x,
                                             n.y,
                                             n.z,
                                             u,
                                             v});
            }
        }

        for (uint32_t z = 0; z < r; ++z)
        {
            for (uint32_t x = 0; x < r; ++x)
            {
                const uint32_t i0 = vertexBase + z * vertsPerSide + x;
                const uint32_t i1 = i0 + 1u;
                const uint32_t i2 = i0 + vertsPerSide;
                const uint32_t i3 = i2 + 1u;
                indices.push_back(i0);
                indices.push_back(i2);
                indices.push_back(i1);
                indices.push_back(i1);
                indices.push_back(i2);
                indices.push_back(i3);
            }
        }

        DrawBatch batch;
        batch.indexOffset = indexBase;
        batch.indexCount = (uint32_t)indices.size() - indexBase;
        batch.sourceTileIndex = sourceTileIndex;
        m_batches.push_back(batch);
    }

    m_tileCount = (uint32_t)tiles.size();
    m_indexCount = (uint32_t)indices.size();
    m_vertexBuffer = nil;
    m_indexBuffer = nil;
    if (!vertices.empty() && !indices.empty())
    {
        m_vertexBuffer = [m_device newBufferWithBytes:vertices.data()
                                               length:vertices.size() * sizeof(VertexPNT)
                                              options:MTLResourceStorageModeShared];
        m_indexBuffer = [m_device newBufferWithBytes:indices.data()
                                              length:indices.size() * sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
    }
}
