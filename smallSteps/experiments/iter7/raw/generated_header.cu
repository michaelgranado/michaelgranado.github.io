#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"
#include <vector> // Needed for std::vector during setup

class CudaRenderer : public CircleRenderer {

private:

    Image* image;
    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;

    // Tile data structures
    int numTilesX, numTilesY, totalTiles;
    int* h_tile_offsets;          // Host temp storage during setup
    int* h_tile_circle_indices;   // Host temp storage during setup
    int* d_tile_offsets;          // Device memory
    int* d_tile_circle_indices;   // Device memory
    int totalTileCircles;         // Size of d_tile_circle_indices

public:

    CudaRenderer();
    virtual ~CudaRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();

    // shadePixel is now defined within cudaRenderer.cu as __device__ __inline__
    // void shadePixel(
    //     int circleIndex,
    //     float pixelCenterX, float pixelCenterY,
    //     float px, float py, float pz,
    //     float* pixelData);
};


#endif
