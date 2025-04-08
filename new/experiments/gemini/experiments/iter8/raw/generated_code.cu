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

    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);

    for (int circleIndex = 0; circleIndex < cuConstRendererParams.numCircles; circleIndex++) {
        int index3 = 3 * circleIndex;

        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float rad = cuConstRendererParams.radius[circleIndex];

        float diffX = p.x - pixelCenterNorm.x;
        float diffY = p.y - pixelCenterNorm.y;
        float pixelDist = diffX * diffX + diffY * diffY;
        float maxDist = rad * rad;

        if (pixelDist <= maxDist) {
            shadePixel(circleIndex, pixelCenterNorm, p, imgPtr);
        }
    }
}
