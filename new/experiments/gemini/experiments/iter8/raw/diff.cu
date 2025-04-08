2,3d1
<     int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
<     int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
5,6c3
<     int imageWidth = cuConstRendererParams.imageWidth;
<     int imageHeight = cuConstRendererParams.imageHeight;
---
>     int index = blockIdx.x * blockDim.x + threadIdx.x;
8c5
<     if (pixelX >= imageWidth || pixelY >= imageHeight)
---
>     if (index >= cuConstRendererParams.numCircles)
11,14c8
<     float invWidth = 1.f / imageWidth;
<     float invHeight = 1.f / imageHeight;
<     float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
<                                         invHeight * (static_cast<float>(pixelY) + 0.5f));
---
>     int index3 = 3 * index;
16c10,11
<     float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
---
>     float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
>     float  rad = cuConstRendererParams.radius[index];
18,19c13,18
<     for (int circleIndex = 0; circleIndex < cuConstRendererParams.numCircles; circleIndex++) {
<         int index3 = 3 * circleIndex;
---
>     short imageWidth = cuConstRendererParams.imageWidth;
>     short imageHeight = cuConstRendererParams.imageHeight;
>     short minX = static_cast<short>(imageWidth * (p.x - rad));
>     short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
>     short minY = static_cast<short>(imageHeight * (p.y - rad));
>     short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;
21,22c20,23
<         float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
<         float rad = cuConstRendererParams.radius[circleIndex];
---
>     short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
>     short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
>     short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
>     short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;
24,27c25,26
<         float diffX = p.x - pixelCenterNorm.x;
<         float diffY = p.y - pixelCenterNorm.y;
<         float pixelDist = diffX * diffX + diffY * diffY;
<         float maxDist = rad * rad;
---
>     float invWidth = 1.f / imageWidth;
>     float invHeight = 1.f / imageHeight;
29,30c28,34
<         if (pixelDist <= maxDist) {
<             shadePixel(circleIndex, pixelCenterNorm, p, imgPtr);
---
>     for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
>         float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
>         for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
>             float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
>                                                  invHeight * (static_cast<float>(pixelY) + 0.5f));
>             shadePixel(index, pixelCenterNorm, p, imgPtr);
>             imgPtr++;
