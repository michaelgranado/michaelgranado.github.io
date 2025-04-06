#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f); // NOTE: Alpha is 1 for background

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a); // NOTE: Alpha is passed in, typically 1.f for background

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
//
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update
        return;
    }
    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    float* radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}


// kernelAdvanceBouncingBalls
//
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}


// *********************************************************************************
// ** NEW KERNEL FOR CORRECTNESS **
// kernelRenderPixelCentric -- (CUDA device code)
//
// Each thread computes the final color for a single pixel.
// It iterates through all circles in order, applying their contributions
// if the pixel center is covered. This ensures atomicity and order.
// *********************************************************************************
__global__ void kernelRenderPixelCentric() {

    // Calculate the pixel coordinates (x, y) this thread is responsible for
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;

    // Check if the calculated pixel coordinates are within the image bounds
    if (pixelX >= imageWidth || pixelY >= imageHeight)
        return;

    // Calculate the normalized coordinates ([0, 1]) of the pixel's center
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float pixelCenterX = invWidth * (static_cast<float>(pixelX) + 0.5f);
    float pixelCenterY = invHeight * (static_cast<float>(pixelY) + 0.5f);
    float2 pixelCenterNorm = make_float2(pixelCenterX, pixelCenterY);

    // Calculate the memory offset for this pixel in the global image buffer
    int pixelOffset = 4 * (pixelY * imageWidth + pixelX);
    float4* imagePtr = (float4*)(&cuConstRendererParams.imageData[pixelOffset]);

    // Initialize the pixel color accumulator with the background color
    // (already written to imageData by clearImage)
    float4 accumulatedColor = *imagePtr;

    // Iterate through all circles in the specified order
    for (int circleIndex = 0; circleIndex < cuConstRendererParams.numCircles; ++circleIndex) {

        int index3 = 3 * circleIndex;

        // Get the current circle's properties (position and radius)
        float3 circlePos = *(float3*)(&cuConstRendererParams.position[index3]);
        float circleRad = cuConstRendererParams.radius[circleIndex];

        // Check if the pixel center is inside the current circle
        float diffX = circlePos.x - pixelCenterNorm.x;
        float diffY = circlePos.y - pixelCenterNorm.y;
        float distSq = diffX * diffX + diffY * diffY;
        float radSq = circleRad * circleRad;

        if (distSq <= radSq) {
            // Pixel center is inside the circle, calculate color contribution and blend

            float3 circleRgb;
            float circleAlpha;

            // Determine circle color and alpha based on scene type
            if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
                const float kCircleMaxAlpha = .5f;
                const float falloffScale = 4.f;

                float normPixelDist = sqrtf(distSq) / circleRad;
                circleRgb = lookupColor(normPixelDist);

                float maxAlpha = .6f + .4f * (1.f - circlePos.z); // Use circle's Z for alpha modulation
                maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // Clamp and scale
                circleAlpha = maxAlpha * expf(-1.f * falloffScale * normPixelDist * normPixelDist);

            } else {
                // Simple case: fixed color and alpha per circle
                circleRgb = *(float3*)&(cuConstRendererParams.color[index3]);
                circleAlpha = .5f; // Default alpha for non-snowflake scenes
            }

            // Perform the blending operation using the "over" operator
            // result = alpha * foreground + (1 - alpha) * background
            // NOTE: We don't update accumulatedColor.w here based on the formula in the handout.
            // The handout formula implies the background already has alpha=1. The resulting alpha is not used later.
            // Let's follow the handout formula precisely for RGB.
            float oneMinusAlpha = 1.f - circleAlpha;
            accumulatedColor.x = circleAlpha * circleRgb.x + oneMinusAlpha * accumulatedColor.x;
            accumulatedColor.y = circleAlpha * circleRgb.y + oneMinusAlpha * accumulatedColor.y;
            accumulatedColor.z = circleAlpha * circleRgb.z + oneMinusAlpha * accumulatedColor.z;
            // The meaning of alpha in the output image isn't clearly defined for blending,
            // but the reference likely keeps it at 1.0. Let's assume the framebuffer alpha stays 1.
            // Or, if we strictly follow the reference `shadePixel` which does `newColor.w = alpha + existingColor.w;`,
            // it seems to accumulate alpha, which is unusual for standard blending.
            // Let's stick to the standard "over" operator's RGB calculation and keep alpha=1 as is typical for opaque framebuffers.
            // accumulatedColor.w = circleAlpha + oneMinusAlpha * accumulatedColor.w; // Correct alpha blending if needed
            accumulatedColor.w = 1.f; // Keep framebuffer opaque - matches clear behavior

        } // End if pixel in circle
    } // End for each circle

    // After processing all circles, write the final accumulated color to global memory
    *imagePtr = accumulatedColor;
}


// *********************************************************************************
// ** ORIGINAL (INCORRECT) KERNEL AND SHADE FUNCTION - NO LONGER USED **
// *********************************************************************************

// shadePixel -- (CUDA device code) - INCORRECT due to non-atomic RMW and order violation potential
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
/*
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION (BUT IS NOT)
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w; // This alpha accumulation is unusual

    // global memory write (RACE CONDITION HERE)
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}
*/

// kernelRenderCircles -- (CUDA device code) - INCORRECT due to atomicity/order violations
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
/*
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            // Call to the incorrect shadePixel function
            // shadePixel(index, pixelCenterNorm, p, imgPtr);
            imgPtr++;
        }
    }
}
*/

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    //printf("Copying image data from device\n"); // Optional: for debugging

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);
    // Consider adding cudaDeviceSynchronize() here and error checking if issues arise
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after memcpy getImage: %s\n", cudaGetErrorString(err));
    }


    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    // Allocate memory on the device
    err = cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    if (err != cudaSuccess) fprintf(stderr, "CUDA Error malloc pos: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
     if (err != cudaSuccess) fprintf(stderr, "CUDA Error malloc vel: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
     if (err != cudaSuccess) fprintf(stderr, "CUDA Error malloc col: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
     if (err != cudaSuccess) fprintf(stderr, "CUDA Error malloc rad: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);
     if (err != cudaSuccess) fprintf(stderr, "CUDA Error malloc img: %s\n", cudaGetErrorString(err));

    // Copy data from host to device
    err = cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
     if (err != cudaSuccess) fprintf(stderr, "CUDA Error memcpy pos: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
     if (err != cudaSuccess) fprintf(stderr, "CUDA Error memcpy vel: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
     if (err != cudaSuccess) fprintf(stderr, "CUDA Error memcpy col: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);
     if (err != cudaSuccess) fprintf(stderr, "CUDA Error memcpy rad: %s\n", cudaGetErrorString(err));

    // Initialize parameters in constant memory.
    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    err = cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));
     if (err != cudaSuccess) fprintf(stderr, "CUDA Error memcpy const params: %s\n", cudaGetErrorString(err));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    err = cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
     if (err != cudaSuccess) fprintf(stderr, "CUDA Error memcpy const noiseX: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
     if (err != cudaSuccess) fprintf(stderr, "CUDA Error memcpy const noiseY: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);
     if (err != cudaSuccess) fprintf(stderr, "CUDA Error memcpy const noiseVal: %s\n", cudaGetErrorString(err));

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    err = cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);
     if (err != cudaSuccess) fprintf(stderr, "CUDA Error memcpy const color ramp: %s\n", cudaGetErrorString(err));

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // Use 2D block dimensions suitable for image processing
    dim3 blockDim(16, 16, 1); // 256 threads per block
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        // Clear to white, fully opaque
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }

    // Check for kernel launch errors after synchronizing
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error during kernel launch (clearImage): %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
     if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error during synchronization (clearImage): %s\n", cudaGetErrorString(err));
    }
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // Keep 1D block/grid for circle-based animation updates
    dim3 blockDim(256, 1, 1); // 256 threads per block
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x, 1, 1);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    } else {
        // No animation for other scenes
        return;
    }

    // Check for kernel launch errors after synchronizing
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error during kernel launch (advanceAnimation): %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error during synchronization (advanceAnimation): %s\n", cudaGetErrorString(err));
    }
}

void
CudaRenderer::render() {

    // Launch the pixel-centric kernel
    // Use 2D block dimensions suitable for image processing
    dim3 blockDim(16, 16, 1); // 256 threads per block
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    // Launch the new, correct kernel
    kernelRenderPixelCentric<<<gridDim, blockDim>>>();

    // Check for kernel launch errors after synchronizing
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error during kernel launch (render): %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error during synchronization (render): %s\n", cudaGetErrorString(err));
    }

    // Original incorrect kernel launch (commented out)
    // dim3 blockDim_orig(256, 1);
    // dim3 gridDim_orig((numCircles + blockDim_orig.x - 1) / blockDim_orig.x);
    // kernelRenderCircles<<<gridDim_orig, blockDim_orig>>>();
    // cudaDeviceSynchronize();
}

// Definition for CudaRenderer::shadePixel (belongs to the class but is unused with the new kernel)
// We keep it to avoid linker errors if it's declared in the header without definition,
// although it's bad practice to keep unused code. Ideally, remove from header too.
void CudaRenderer::shadePixel(int, float, float, float, float, float, float*) {
    // This function is intentionally left empty as it's replaced by
    // the logic inside kernelRenderPixelCentric.
}
