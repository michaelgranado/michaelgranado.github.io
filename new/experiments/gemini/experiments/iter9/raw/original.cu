__global__ void kernelRenderCircles() {
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;

    if (pixelX >= imageWidth || pixelY >= imageHeight)
        return;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));

    float tileL = invWidth * (blockIdx.x * blockDim.x);
    float tileR = invWidth * ((blockIdx.x + 1) * blockDim.x);
    float tileB = invHeight * (blockIdx.y * blockDim.y);
    float tileT = invHeight * ((blockIdx.y + 1) * blockDim.y);

    int pixelOffset = 4 * (pixelY * imageWidth + pixelX);
    float4* imgPtrGlobal = (float4*)(&cuConstRendererParams.imageData[pixelOffset]);

    float4 accumulatedColor = *imgPtrGlobal;

    for (int circleIndex = 0; circleIndex < cuConstRendererParams.numCircles; circleIndex++) {
        int index3 = 3 * circleIndex;

        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float rad = cuConstRendererParams.radius[circleIndex];

        if (circleInBoxConservative(p.x, p.y, rad, tileL, tileR, tileT, tileB)) {
		shadePixel(circleIndex, pixelCenterNorm, p, accumulatedColor);
        }
    } 

    *imgPtrGlobal = accumulatedColor;
}
