2,8c2,3
<     __shared__ int tileIntersectingIndices[MAX_CIRCLES_PER_TILE];
<     __shared__ unsigned int isIntersecting[SCAN_BLOCK_DIM];
<     __shared__ unsigned int scanOutput[SCAN_BLOCK_DIM];
<     __shared__ unsigned int scanScratch[2 * SCAN_BLOCK_DIM];
<     __shared__ int currentTileListSize;
<     __shared__ int chunkIntersectCountShared;
<     __shared__ int chunkBaseOffsetShared;
---
>     int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
>     int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
10,16d4
<     int linearThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;
< 
<     if (linearThreadIdx == 0) {
<         currentTileListSize = 0;
<     }
<     __syncthreads();
< 
18a7,10
> 
>     if (pixelX >= imageWidth || pixelY >= imageHeight)
>         return;
> 
22,25c14,15
<     float tileL = invWidth * (blockIdx.x * RENDER_BLOCK_DIM_X);
<     float tileR = invWidth * ((blockIdx.x + 1) * RENDER_BLOCK_DIM_X);
<     float tileB = invHeight * (blockIdx.y * RENDER_BLOCK_DIM_Y);
<     float tileT = invHeight * ((blockIdx.y + 1) * RENDER_BLOCK_DIM_Y);
---
>     float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
>                                          invHeight * (static_cast<float>(pixelY) + 0.5f));
27c17,20
<     int totalCircles = cuConstRendererParams.numCircles;
---
>     float tileL = invWidth * (blockIdx.x * blockDim.x);
>     float tileR = invWidth * ((blockIdx.x + 1) * blockDim.x);
>     float tileB = invHeight * (blockIdx.y * blockDim.y);
>     float tileT = invHeight * ((blockIdx.y + 1) * blockDim.y);
29c22,23
<     for (int chunkBaseCircleIdx = 0; chunkBaseCircleIdx < totalCircles; chunkBaseCircleIdx += SCAN_BLOCK_DIM) {
---
>     int pixelOffset = 4 * (pixelY * imageWidth + pixelX);
>     float4* imgPtrGlobal = (float4*)(&cuConstRendererParams.imageData[pixelOffset]);
31,32c25
<         int circleIdx = chunkBaseCircleIdx + linearThreadIdx;
<         bool intersectsTile = false;
---
>     float4 accumulatedColor = *imgPtrGlobal;
34,36c27,28
<         if (circleIdx < totalCircles) {
<             float3 p = *(float3*)(&cuConstRendererParams.position[3 * circleIdx]);
<             float rad = cuConstRendererParams.radius[circleIdx];
---
>     for (int circleIndex = 0; circleIndex < cuConstRendererParams.numCircles; circleIndex++) {
>         int index3 = 3 * circleIndex;
38,39c30,31
<             intersectsTile = circleInBoxConservative(p.x, p.y, rad, tileL, tileR, tileT, tileB);
<         }
---
>         float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
>         float rad = cuConstRendererParams.radius[circleIndex];
41,50c33,34
<         isIntersecting[linearThreadIdx] = intersectsTile ? 1 : 0;
< 
<         __syncthreads();
< 
<         sharedMemExclusiveScan(linearThreadIdx, isIntersecting, scanOutput, scanScratch, SCAN_BLOCK_DIM);
< 
<         __syncthreads();
< 
<         if (linearThreadIdx == SCAN_BLOCK_DIM - 1) {
<             chunkIntersectCountShared = scanOutput[SCAN_BLOCK_DIM - 1] + isIntersecting[SCAN_BLOCK_DIM - 1];
---
>         if (circleInBoxConservative(p.x, p.y, rad, tileL, tileR, tileT, tileB)) {
> 		shadePixel(circleIndex, pixelCenterNorm, p, accumulatedColor);
51a36
>     } 
53,105c38
<         __syncthreads();
< 
<         if (chunkIntersectCountShared > 0) {
<             if (linearThreadIdx == 0) {
<                 chunkBaseOffsetShared = atomicAdd(¤tTileListSize, chunkIntersectCountShared);
<             }
< 
<             __syncthreads();
< 
<             if (intersectsTile) {
<                 int writeIdx = chunkBaseOffsetShared + scanOutput[linearThreadIdx];
< 
<                 if (writeIdx < MAX_CIRCLES_PER_TILE) {
<                     tileIntersectingIndices[writeIdx] = circleIdx;
<                 } else {
<                     if (linearThreadIdx == 0 && chunkBaseOffsetShared + scanOutput[linearThreadIdx] == MAX_CIRCLES_PER_TILE) {
<                          printf("Warning: MAX_CIRCLES_PER_TILE (%d) exceeded in block (%d, %d). Some circles may be missed.\n",
<                                 MAX_CIRCLES_PER_TILE, blockIdx.x, blockIdx.y);
<                     }
<                 }
<             }
<         }
<         __syncthreads();
<     }
< 
<     int pixelX = blockIdx.x * RENDER_BLOCK_DIM_X + threadIdx.x;
<     int pixelY = blockIdx.y * RENDER_BLOCK_DIM_Y + threadIdx.y;
< 
<     if (pixelX < imageWidth && pixelY < imageHeight) {
< 
<         float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
<                                              invHeight * (static_cast<float>(pixelY) + 0.5f));
< 
<         int pixelOffset = 4 * (pixelY * imageWidth + pixelX);
<         float4* imgPtrGlobal = (float4*)(&cuConstRendererParams.imageData[pixelOffset]);
< 
<         float4 accumulatedColor = *imgPtrGlobal;
< 
<         int numCirclesInTile = currentTileListSize;
<         if (numCirclesInTile > MAX_CIRCLES_PER_TILE) {
<              numCirclesInTile = MAX_CIRCLES_PER_TILE;
<         }
< 
<         for (int i = 0; i < numCirclesInTile; ++i) {
<             int globalCircleIndex = tileIntersectingIndices[i];
< 
<             float3 p = *(float3*)(&cuConstRendererParams.position[3 * globalCircleIndex]);
< 
<             shadePixel(globalCircleIndex, pixelCenterNorm, p, accumulatedColor);
<         }
< 
<         *imgPtrGlobal = accumulatedColor;
<     }
---
>     *imgPtrGlobal = accumulatedColor;
