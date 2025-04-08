13d12
< 
15c14
<                                          invHeight * (static_cast<float>(pixelY) + 0.5f));
---
>                                         invHeight * (static_cast<float>(pixelY) + 0.5f));
17,20c16
<     float tileL = invWidth * (blockIdx.x * blockDim.x);
<     float tileR = invWidth * ((blockIdx.x + 1) * blockDim.x);
<     float tileB = invHeight * (blockIdx.y * blockDim.y);
<     float tileT = invHeight * ((blockIdx.y + 1) * blockDim.y);
---
>     float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
22,26d17
<     int pixelOffset = 4 * (pixelY * imageWidth + pixelX);
<     float4* imgPtrGlobal = (float4*)(&cuConstRendererParams.imageData[pixelOffset]);
< 
<     float4 accumulatedColor = *imgPtrGlobal;
< 
33,36c24,27
<         if (circleInBoxConservative(p.x, p.y, rad, tileL, tileR, tileT, tileB)) {
< 		shadePixel(circleIndex, pixelCenterNorm, p, accumulatedColor);
<         }
<     } 
---
>         float diffX = p.x - pixelCenterNorm.x;
>         float diffY = p.y - pixelCenterNorm.y;
>         float pixelDist = diffX * diffX + diffY * diffY;
>         float maxDist = rad * rad;
38c29,32
<     *imgPtrGlobal = accumulatedColor;
---
>         if (pixelDist <= maxDist) {
>             shadePixel(circleIndex, pixelCenterNorm, p, imgPtr);
>         }
>     }
