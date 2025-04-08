1a2,3
>     int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
>     int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
3,9d4
<     __shared__ float3 s_positions[BLOCK_SIZE];
<     __shared__ float  s_radii[BLOCK_SIZE];
<     __shared__ float3 s_colors[BLOCK_SIZE];
< 
<     int pixelX = blockIdx.x * TILE_SIZE_X + threadIdx.x;
<     int pixelY = blockIdx.y * TILE_SIZE_Y + threadIdx.y;
< 
16,17d10
<     int tid = threadIdx.y * TILE_SIZE_X + threadIdx.x;
< 
19a13
> 
23,26c17,20
<     float tileL = invWidth * (blockIdx.x * TILE_SIZE_X);
<     float tileR = invWidth * ((blockIdx.x + 1) * TILE_SIZE_X);
<     float tileB = invHeight * (blockIdx.y * TILE_SIZE_Y);
<     float tileT = invHeight * ((blockIdx.y + 1) * TILE_SIZE_Y);
---
>     float tileL = invWidth * (blockIdx.x * blockDim.x);
>     float tileR = invWidth * ((blockIdx.x + 1) * blockDim.x);
>     float tileB = invHeight * (blockIdx.y * blockDim.y);
>     float tileT = invHeight * ((blockIdx.y + 1) * blockDim.y);
33,34c27,28
<     bool loadSharedColor = (cuConstRendererParams.sceneName != SNOWFLAKES &&
<                             cuConstRendererParams.sceneName != SNOWFLAKES_SINGLE_FRAME);
---
>     for (int circleIndex = 0; circleIndex < cuConstRendererParams.numCircles; circleIndex++) {
>         int index3 = 3 * circleIndex;
36,37c30,31
<     int numCircles = cuConstRendererParams.numCircles;
<     for (int chunkBase = 0; chunkBase < numCircles; chunkBase += BLOCK_SIZE) {
---
>         float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
>         float rad = cuConstRendererParams.radius[circleIndex];
39,47c33,34
<         int circleIdxGlobal = chunkBase + tid;
< 
<         if (circleIdxGlobal < numCircles) {
<             s_positions[tid] = *(float3*)&(cuConstRendererParams.position[3 * circleIdxGlobal]);
<             s_radii[tid] = cuConstRendererParams.radius[circleIdxGlobal];
< 
<             if (loadSharedColor) {
<                 s_colors[tid] = *(float3*)&(cuConstRendererParams.color[3 * circleIdxGlobal]);
<             }
---
>         if (circleInBoxConservative(p.x, p.y, rad, tileL, tileR, tileT, tileB)) {
> 		shadePixel(circleIndex, pixelCenterNorm, p, accumulatedColor);
48a36
>     } 
50,69d37
<         __syncthreads();
< 
<         int numCirclesInChunk = min(BLOCK_SIZE, numCircles - chunkBase);
< 
<         for (int i = 0; i < numCirclesInChunk; ++i) {
<             float3 p_shared = s_positions[i];
<             float rad_shared = s_radii[i];
<             float3 color_shared = loadSharedColor ? s_colors[i] : make_float3(0.f, 0.f, 0.f);
< 
<             int currentCircleIdxGlobal = chunkBase + i;
< 
<             if (circleInBoxConservative(p_shared.x, p_shared.y, rad_shared, tileL, tileR, tileT, tileB)) {
<                 shadePixelOptimized(currentCircleIdxGlobal, pixelCenterNorm, p_shared, rad_shared, color_shared, accumulatedColor);
<             }
<         }
< 
<         __syncthreads();
< 
<     }
< 
