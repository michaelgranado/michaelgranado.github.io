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

// Use cudaCheckError wrapper for debugging memory allocation issues
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
if (code != cudaSuccess)
{
  fprintf(stderr, "CUDA Error: %s at %s:%d\n",
    cudaGetErrorString(code), file, line);
  if (abort) exit(code);
}
}

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
#include "circleBoxTest.cu_inl"

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
    float4 value = make_float4(shade, shade, shade, 1.f); // Initialize alpha to 1? Or 0? Let's assume 0 for accumulation. Revisit if needed.
                                                            // The reference clear sets alpha to 1. Let's keep it 1.
                                                            // The accumulation logic adds alpha, so maybe background should be alpha=0?
                                                            // Let's stick to the original clear logic for now.
                                                            // The original shadePixel used existingColor.w, implying background alpha matters.
                                                            // Let's keep background alpha as 1.0f as in original kernelClearImage.


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
    float4 value = make_float4(r, g, b, a);

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


// kernelRenderCirclesOrdered -- (CUDA device code)
//
// Each thread renders a single pixel, processing all circles in order.
// Accumulates color locally in registers and writes once to global memory.
__global__ void kernelRenderCirclesOrdered() {
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;

    // Check if this thread's pixel is within the image bounds
    if (pixelX >= imageWidth || pixelY >= imageHeight)
        return;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));

    // Get a pointer to this pixel's color data in global memory
    int pixelOffset = 4 * (pixelY * imageWidth + pixelX);
    float4* imgPtrGlobal = (float4*)(&cuConstRendererParams.imageData[pixelOffset]);

    // Initialize accumulated color locally (in registers)
    // Read the initial background color for this pixel (set by clearImage)
    float4 accumulatedColor = *imgPtrGlobal;

    // Process all circles in order
    for (int circleIndex = 0; circleIndex < cuConstRendererParams.numCircles; circleIndex++) {
        int index3 = 3 * circleIndex;

        // Read circle position and radius
        // Optimization: Read position as float3 directly
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float rad = cuConstRendererParams.radius[circleIndex];

        // Compute the distance squared from pixel center to circle center
        float diffX = p.x - pixelCenterNorm.x;
        float diffY = p.y - pixelCenterNorm.y;
        float pixelDistSq = diffX * diffX + diffY * diffY;
        float maxDistSq = rad * rad;

        // If pixel is within the circle's radius squared, compute color and blend locally
        if (pixelDistSq <= maxDistSq) {
            // --- Inlined shading and blending logic ---
            float3 rgb;
            float alpha;

            // Determine base color and alpha based on scene type
            // Optimization: Hoist the scene check outside the loop if possible?
            // No, color/alpha depend on circleIndex and p.z which change in loop.
            if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
                const float kCircleMaxAlpha = .5f;
                const float falloffScale = 4.f;

                // Avoid sqrt if possible, but lookupColor needs normalized distance
                float pixelDist = sqrtf(pixelDistSq); // Need sqrt here
                float normPixelDist = pixelDist / rad; // Handle rad=0? Assume rad > 0
                rgb = lookupColor(normPixelDist);

                float maxAlpha = .6f + .4f * (1.f-p.z);
                maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value

                // Use pixelDistSq for exp calculation
                alpha = maxAlpha * expf(-1.f * falloffScale * normPixelDist * normPixelDist); // normPixelDist^2 = pixelDistSq / maxDistSq
                // alpha = maxAlpha * expf(-1.f * falloffScale * pixelDistSq / maxDistSq); // Equivalent, maybe faster? Let's use original for safety.

            } else {
                // Simple: each circle has an assigned color
                // Optimization: Read color as float3 directly
                rgb = *(float3*)&(cuConstRendererParams.color[index3]);
                // Use the alpha component from the color data if available?
                // The original code hardcoded alpha = 0.5f here. Let's stick to that.
                alpha = .5f;
            }

            // Blend with the *accumulated* color using standard over operator
            // result = foreground * alpha + background * (1 - alpha)
            float oneMinusAlpha = 1.f - alpha;
            accumulatedColor.x = alpha * rgb.x + oneMinusAlpha * accumulatedColor.x;
            accumulatedColor.y = alpha * rgb.y + oneMinusAlpha * accumulatedColor.y;
            accumulatedColor.z = alpha * rgb.z + oneMinusAlpha * accumulatedColor.z;

            // Accumulate alpha according to the original logic (summation)
            // This seems non-standard for blending but preserves original behavior.
            // Standard alpha blending: accumulatedColor.w = alpha + oneMinusAlpha * accumulatedColor.w;
            // Original logic:
            accumulatedColor.w = alpha + accumulatedColor.w;

            // --- End inlined shading and blending logic ---
        }
    }

    // Write the final accumulated color to global memory ONCE
    *imgPtrGlobal = accumulatedColor;
}


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

    // printf("Copying image data from device\n"); // Reduce print noise

    cudaError_t err = cudaMemcpy(image->data,
                                 cudaDeviceImageData,
                                 sizeof(float) * 4 * image->width * image->height,
                                 cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error copying image data to host: %s\n", cudaGetErrorString(err));
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


    cudaCheckError(cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles));
    cudaCheckError(cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles));
    cudaCheckError(cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles));
    cudaCheckError(cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles));
    cudaCheckError(cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height));


    cudaCheckError(cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice));


    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

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

    cudaCheckError(cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants)));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaCheckError(cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256));
    cudaCheckError(cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256));
    cudaCheckError(cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256));

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaCheckError(cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE));

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

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1); // 256 threads
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        // Clear with alpha = 0, so accumulation starts correctly?
        // The original code cleared to alpha=1. Let's keep that.
        // The accumulation `accumulatedColor.w = alpha + accumulatedColor.w;`
        // assumes the initial w value matters.
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaError_t err = cudaGetLastError(); // Check for errors after kernel launch
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after clearImage kernel launch: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); // Wait for clear to complete
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }

    cudaError_t err = cudaGetLastError(); // Check for errors after kernel launch
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after advanceAnimation kernel launch: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize(); // Wait for animation update to complete
}

// kernelRenderCirclesTiledOptimized -- (CUDA device code)
//
// Each thread renders a single pixel.
// Uses tile-based culling to skip circles far from the pixel's tile.
// Cooperatively loads batches of circle data into shared memory to reduce global memory reads.
// Processes remaining relevant circles in order for the pixel.
// Accumulates color locally in registers and writes once to global memory.
//
// TILE_DIM_X and TILE_DIM_Y should match blockDim.x and blockDim.y
#define TILE_DIM_X 16
#define TILE_DIM_Y 16
#define CIRCLE_BATCH_SIZE (TILE_DIM_X * TILE_DIM_Y) // 256 threads load 256 circles per batch

__global__ void kernelRenderCirclesTiledOptimized() {

    // Shared memory for caching one batch of circle data
    __shared__ float3 s_positions[CIRCLE_BATCH_SIZE];
    __shared__ float  s_radii[CIRCLE_BATCH_SIZE];
    // Note: We could cache color here too, but let's fetch it only when needed
    // to save shared memory, especially since color fetch depends on scene type.
    // __shared__ float3 s_colors[CIRCLE_BATCH_SIZE];

    // Calculate pixel coordinates for this thread
    int pixelX = blockIdx.x * TILE_DIM_X + threadIdx.x;
    int pixelY = blockIdx.y * TILE_DIM_Y + threadIdx.y;

    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;

    // Check if this thread's pixel is within the image bounds
    if (pixelX >= imageWidth || pixelY >= imageHeight)
        return;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // Calculate normalized coordinates for the pixel center
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));

    // Calculate the normalized boundaries of this thread's tile (block)
    // These are constant for all threads in the block
    float tileL = invWidth * (blockIdx.x * TILE_DIM_X);
    float tileR = invWidth * ((blockIdx.x + 1) * TILE_DIM_X);
    float tileB = invHeight * (blockIdx.y * TILE_DIM_Y);
    float tileT = invHeight * ((blockIdx.y + 1) * TILE_DIM_Y);

    // Get a pointer to this pixel's color data in global memory
    int pixelOffset = 4 * (pixelY * imageWidth + pixelX);
    float4* imgPtrGlobal = (float4*)(&cuConstRendererParams.imageData[pixelOffset]);

    // Initialize accumulated color locally (in registers)
    // Read the initial background color for this pixel (set by clearImage)
    float4 accumulatedColor = *imgPtrGlobal;

    // Thread index within the block (linearized)
    int threadLinearIndex = threadIdx.y * TILE_DIM_X + threadIdx.x;

    // Process circles in batches
    int numCircles = cuConstRendererParams.numCircles;
    for (int batchStart = 0; batchStart < numCircles; batchStart += CIRCLE_BATCH_SIZE) {

        // Phase 1: Cooperatively load circle data for the current batch into shared memory
        int circleLoadIndex = batchStart + threadLinearIndex;
        if (threadLinearIndex < CIRCLE_BATCH_SIZE && circleLoadIndex < numCircles) {
             // Load position (float3) and radius (float)
             s_positions[threadLinearIndex] = *(float3*)(&cuConstRendererParams.position[3 * circleLoadIndex]);
             s_radii[threadLinearIndex] = cuConstRendererParams.radius[circleLoadIndex];
             // If caching color:
             // s_colors[threadLinearIndex] = *(float3*)(&cuConstRendererParams.color[3 * circleLoadIndex]);
        }

        // Synchronize to ensure all threads in the block have finished loading the batch
        __syncthreads();

        // Phase 2: Process the batch of circles (reading from shared memory)
        // Iterate through the circles *in this batch*
        int numCirclesInBatch = min(CIRCLE_BATCH_SIZE, numCircles - batchStart);
        for (int i = 0; i < numCirclesInBatch; ++i) {
            int currentCircleIndex = batchStart + i; // Global index of the circle

            // Read position and radius from shared memory
            float3 p_s = s_positions[i];
            float rad_s = s_radii[i];

            // *** Tile Culling Check *** (using shared memory data)
            // Use conservative test: if circle doesn't overlap the tile's bounding box, skip it
            if (circleInBoxConservative(p_s.x, p_s.y, rad_s, tileL, tileR, tileT, tileB)) {

                // *** Pixel Distance Check *** (using shared memory data)
                // Compute the distance squared from pixel center to circle center
                float diffX = p_s.x - pixelCenterNorm.x;
                float diffY = p_s.y - pixelCenterNorm.y;
                float pixelDistSq = diffX * diffX + diffY * diffY;
                float maxDistSq = rad_s * rad_s;

                // If pixel is within the circle's radius squared, compute color and blend locally
                if (pixelDistSq <= maxDistSq) {
                    // --- Inlined shading and blending logic ---
                    float3 rgb;
                    float alpha;

                    // Determine base color and alpha based on scene type
                    // Fetch color from global memory *only* when needed
                    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
                        const float kCircleMaxAlpha = .5f;
                        const float falloffScale = 4.f;

                        float pixelDist = sqrtf(pixelDistSq); // Need sqrt for lookupColor
                        float normPixelDist = (rad_s > 1e-6f) ? (pixelDist / rad_s) : 0.f; // Avoid division by zero/small radius
                        rgb = lookupColor(normPixelDist);

                        float maxAlpha = .6f + .4f * (1.f - p_s.z); // Use z from shared pos
                        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value

                        alpha = maxAlpha * expf(-1.f * falloffScale * normPixelDist * normPixelDist);

                    } else {
                        // Simple: each circle has an assigned color - fetch from global
                        rgb = *(float3*)&(cuConstRendererParams.color[3 * currentCircleIndex]);
                        alpha = .5f; // Use fixed alpha as in previous version
                    }

                    // Blend with the *accumulated* color using standard over operator
                    float oneMinusAlpha = 1.f - alpha;
                    accumulatedColor.x = alpha * rgb.x + oneMinusAlpha * accumulatedColor.x;
                    accumulatedColor.y = alpha * rgb.y + oneMinusAlpha * accumulatedColor.y;
                    accumulatedColor.z = alpha * rgb.z + oneMinusAlpha * accumulatedColor.z;
                    // Keep original alpha accumulation for correctness check comparison
                    accumulatedColor.w += alpha;
                    // --- End inlined shading and blending logic ---
                }
            } // End of tile culling check
        } // End loop over circles in the batch

        // Synchronize before loading the next batch
        // Ensures all reads from shared memory for this batch are complete
        __syncthreads();

    } // End loop over batches

    // Write the final accumulated color to global memory ONCE
    *imgPtrGlobal = accumulatedColor;
}

void
CudaRenderer::render() {
    // Set up a 2D grid of threads, where each thread processes one pixel
    // Use TILE_DIM_X x TILE_DIM_Y blocks (e.g., 16x16 = 256 threads)
    dim3 blockDim(TILE_DIM_X, TILE_DIM_Y);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    kernelRenderCirclesTiledOptimized<<<gridDim, blockDim>>>();

    cudaError_t err = cudaGetLastError(); // Check for errors after kernel launch
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after render kernel launch: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize(); // Wait for rendering to complete
}

// shadePixel function is no longer needed as its logic is inlined into kernelRenderCirclesOrdered
/*
void
CudaRenderer::shadePixel(
    int circleIndex,
    float pixelCenterX, float pixelCenterY,
    float px, float py, float pz,
    float* pixelData) {
    // This function is conceptually replaced by the logic inside kernelRenderCirclesOrdered
}
*/
