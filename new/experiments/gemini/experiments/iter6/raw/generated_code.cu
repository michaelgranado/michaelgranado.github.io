__global__ void kernelRenderCirclesTiled() {

    __shared__ float3 s_positions[BLOCK_SIZE];
    __shared__ float  s_radii[BLOCK_SIZE];
    __shared__ float3 s_colors[BLOCK_SIZE];

    int pixelX = blockIdx.x * TILE_SIZE_X + threadIdx.x;
    int pixelY = blockIdx.y * TILE_SIZE_Y + threadIdx.y;

    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;

    if (pixelX >= imageWidth || pixelY >= imageHeight)
        return;

    int tid = threadIdx.y * TILE_SIZE_X + threadIdx.x;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));

    float tileL = invWidth * (blockIdx.x * TILE_SIZE_X);
    float tileR = invWidth * ((blockIdx.x + 1) * TILE_SIZE_X);
    float tileB = invHeight * (blockIdx.y * TILE_SIZE_Y);
    float tileT = invHeight * ((blockIdx.y + 1) * TILE_SIZE_Y);

    int pixelOffset = 4 * (pixelY * imageWidth + pixelX);
    float4* imgPtrGlobal = (float4*)(&cuConstRendererParams.imageData[pixelOffset]);

    float4 accumulatedColor = *imgPtrGlobal;

    bool loadSharedColor = (cuConstRendererParams.sceneName != SNOWFLAKES &&
                            cuConstRendererParams.sceneName != SNOWFLAKES_SINGLE_FRAME);

    int numCircles = cuConstRendererParams.numCircles;
    for (int chunkBase = 0; chunkBase < numCircles; chunkBase += BLOCK_SIZE) {

        int circleIdxGlobal = chunkBase + tid;

        if (circleIdxGlobal < numCircles) {
            s_positions[tid] = *(float3*)&(cuConstRendererParams.position[3 * circleIdxGlobal]);
            s_radii[tid] = cuConstRendererParams.radius[circleIdxGlobal];

            if (loadSharedColor) {
                s_colors[tid] = *(float3*)&(cuConstRendererParams.color[3 * circleIdxGlobal]);
            }
        }

        __syncthreads();

        int numCirclesInChunk = min(BLOCK_SIZE, numCircles - chunkBase);

        for (int i = 0; i < numCirclesInChunk; ++i) {
            float3 p_shared = s_positions[i];
            float rad_shared = s_radii[i];
            float3 color_shared = loadSharedColor ? s_colors[i] : make_float3(0.f, 0.f, 0.f);

            int currentCircleIdxGlobal = chunkBase + i;

            if (circleInBoxConservative(p_shared.x, p_shared.y, rad_shared, tileL, tileR, tileT, tileB)) {
                shadePixelOptimized(currentCircleIdxGlobal, pixelCenterNorm, p_shared, rad_shared, color_shared, accumulatedColor);
            }
        }

        __syncthreads();

    }

    *imgPtrGlobal = accumulatedColor;
}
