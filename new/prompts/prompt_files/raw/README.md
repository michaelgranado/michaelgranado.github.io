# Assignment 3: A Simple CUDA Renderer

## Overview

In this assignment you will write a parallel renderer in CUDA that draws colored circles.
While this renderer is very simple, parallelizing the renderer will require you to design and implement data structures
that can be efficiently constructed and manipulated in parallel. This is a challenging
assignment so you are advised to start early. **Seriously, you are advised to start early.** Good luck!

## Part 3: A Simple Circle Renderer (85 pts)

Now for the real show!

The directory `/render` of the assignment starter code contains an implementation of renderer that draws colored
circles. Build the code, and run the render with the following command line: `./render -r cpuref rgb`. The program will output an image `output_0000.ppm` containing three circles. Now run the renderer with the command line `./render -r cpuref snow`. Now the output image will be falling snow. PPM images can be viewed directly on OSX via preview. For windows you might need to download a viewer.

Note: you can also use the `-i` option to send renderer output to the display instead of a file. (In the case of snow, you'll see an animation of falling snow.) However, to use interactive mode you'll need to be able to setup X-windows forwarding to your local machine. ([This reference](http://atechyblog.blogspot.com/2014/12/google-cloud-compute-x11-forwarding.html) or [this reference](https://stackoverflow.com/questions/25521486/x11-forwarding-from-debian-on-google-compute-engine) may help.)

The assignment starter code contains two versions of the renderer: a sequential, single-threaded C++
reference implementation, implemented in `refRenderer.cpp`, and an _incorrect_ parallel CUDA implementation in
`cudaRenderer.cu`.

### Renderer Overview

We encourage you to familiarize yourself with the structure of the renderer codebase by inspecting the reference
implementation in `refRenderer.cpp`. The method `setup` is called prior to rendering the first frame. In your CUDA-accelerated
renderer, this method will likely contain all your renderer initialization code (allocating buffers, etc). `render`
is called each frame and is responsible for drawing all circles into the output image. The other main function of
the renderer, `advanceAnimation`, is also invoked once per frame. It updates circle positions and velocities.
You will not need to modify `advanceAnimation` in this assignment.

The renderer accepts an array of circles (3D position, velocity, radius, color) as input. The basic sequential
algorithm for rendering each frame is:

    Clear image
    for each circle
        update position and velocity
    for each circle
        compute screen bounding box
        for all pixels in bounding box
            compute pixel center point
            if center point is within the circle
                compute color of circle at point
                blend contribution of circle into image for this pixel

The figure below illustrates the basic algorithm for computing circle-pixel coverage using point-in-circle tests. Notice that a circle contributes color to an output pixel only if the pixel's center lies within the circle.

![Point in circle test](handout/point_in_circle.jpg?raw=true "A simple algorithm for computing the contribution of a circle to the output image: All pixels within the circle's bounding box are tested for coverage. For each pixel in the bounding box, the pixel is considered to be covered by the circle if its center point (black dots) is contained within the circle. Pixel centers that are inside the circle are colored red. The circle's contribution to the image will be computed only for covered pixels.")

An important detail of the renderer is that it renders **semi-transparent** circles. Therefore, the color of any one pixel is not the color of a single circle, but the result of blending the contributions of all the semi-transparent circles overlapping the pixel (note the "blend contribution" part of the pseudocode above). The renderer represents the color of a circle via a 4-tuple of red (R), green (G), blue (B), and opacity (alpha) values (RGBA). Alpha = 1 corresponds to a fully opaque circle. Alpha = 0 corresponds to a fully transparent circle. To draw a semi-transparent circle with color `(C_r, C_g, C_b, C_alpha)` on top of a pixel with color `(P_r, P_g, P_b)`, the renderer uses the following math:

<pre>
   result_r = C_alpha * C_r + (1.0 - C_alpha) * P_r
   result_g = C_alpha * C_g + (1.0 - C_alpha) * P_g
   result_b = C_alpha * C_b + (1.0 - C_alpha) * P_b
</pre>

Notice that composition is not commutative (object X over Y does not look the same as object Y over X), so it's important that the render draw circles in a manner that follows the order they are provided by the application. (You can assume the application provides the circles in depth order.) For example, consider the two images below where a blue circle is drawn OVER a green circle which is drawn OVER a red circle. In the image on the left, the circles are drawn into the output image in the correct order. In the image on the right, the circles are drawn in a different order, and the output image does not look correct.

![Ordering](handout/order.jpg?raw=true "The renderer must be careful to generate output that is the same as what is generated when sequentially drawing all circles in the order provided by the application.")

### CUDA Renderer

After familiarizing yourself with the circle rendering algorithm as implemented in the reference code, now
study the CUDA implementation of the renderer provided in `cudaRenderer.cu`. You can run the CUDA
implementation of the renderer using the `--renderer cuda (or -r cuda)` cuda program option.

The provided CUDA implementation parallelizes computation across all input circles, assigning one circle to
each CUDA thread. While this CUDA implementation is a complete implementation of the mathematics of
a circle renderer, it contains several major errors that you will fix in this assignment. Specifically: the current
implementation does not ensure image update is an atomic operation and it does not preserve the required
order of image updates (the ordering requirement will be described below).

### Renderer Requirements

Your parallel CUDA renderer implementation must maintain two invariants that are preserved trivially in
the sequential implementation.

1. **Atomicity:** All image update operations must be atomic. The critical region includes reading the
   four 32-bit floating-point values (the pixel's rgba color), blending the contribution of the current circle with
   the current image value, and then writing the pixel's color back to memory.
2. **Order:** Your renderer must perform updates to an image pixel in _circle input order_. That is, if
   circle 1 and circle 2 both contribute to pixel P, any image updates to P due to circle 1 must be applied to the
   image before updates to P due to circle 2. As discussed above, preserving the ordering requirement
   allows for correct rendering of transparent circles. (It has a number of other benefits for graphics
   systems. If curious, talk to Kayvon.) **A key observation is that the definition of order only specifies the order of updates to the same pixel.** Thus, as shown below, there are no ordering requirements between circles that do not contribute to the same pixel. These circles can be processed independently.

![Dependencies](handout/dependencies.jpg?raw=true "The contributions of circles 1, 2, and 3 must be applied to overlapped pixels in the order the circles are provided to the renderer.")

Since the provided CUDA implementation does not satisfy either of these requirements, the result of not correctly
respecting order or atomicity can be seen by running the CUDA renderer implementation on the rgb and circles scenes.
You will see horizontal streaks through the resulting images, as shown below. These streaks will change with each frame.

![Order_errors](handout/bug_example.jpg?raw=true "Errors in the output due to lack of atomicity in frame-buffer update (notice streaks in bottom of image).")

### What You Need To Do

**Your job is to write the fastest, correct CUDA renderer implementation you can**. You may take any approach you
see fit, but your renderer must adhere to the atomicity and order requirements specified above. A solution that does not meet both requirements will be given no more than 12 points on part 3 of the assignment. We have already given you such a solution!

A good place to start would be to read through `cudaRenderer.cu` and convince yourself that it _does not_ meet the correctness requirement. In particular, look at how `CudaRenderer:render` launches the CUDA kernel `kernelRenderCircles`. (`kernelRenderCircles` is where all the work happens.) To visually see the effect of violation of above two requirements, compile the program with `make`. Then run `./render -r cuda rand10k` which should display the image with 10K circles, shown in the bottom row of the image above. Compare this (incorrect) image with the image generated by sequential code by running `./render -r cpuref rand10k`.

We recommend that you:

1. First rewrite the CUDA starter code implementation so that it is logically correct when running in parallel (we recommend an approach that does not require locks or synchronization)
2. Then determine what performance problem is with your solution.
3. At this point the real thinking on the assignment begins... (Hint: the circle-intersects-box tests provided to you in `circleBoxTest.cu_inl` are your friend. You are encouraged to use these subroutines.)

Following are commandline options to `./render`:

```
Usage: ./render [options] scenename
Valid scenenames are: rgb, rgby, rand10k, rand100k, rand1M, biglittle, littlebig, pattern, micro2M,
                      bouncingballs, fireworks, hypnosis, snow, snowsingle
Program Options:
  -r  --renderer <cpuref/cuda>  Select renderer: ref or cuda (default=cuda)
  -s  --size  <INT>             Rendered image size: <INT>x<INT> pixels (default=1024)
  -b  --bench <START:END>       Run for frames [START,END) (default=[0,1))
  -c  --check                   Check correctness of CUDA output against CPU reference
  -i  --interactive             Render output to interactive display
  -f  --file  <FILENAME>        Output file name (FILENAME_xxxx.ppm) (default=output)
  -?  --help                    This message
```

**Checker code:** To detect correctness of the program, `render` has a convenient `--check` option. This option runs the sequential version of the reference CPU renderer along with your CUDA renderer and then compares the resulting images to ensure correctness. The time taken by your CUDA renderer implementation is also printed.

We provide a total of eight circle datasets you will be graded on. However, in order to receive full credit, your code must pass all of our correctness-tests. To check the correctness and performance score of your code, run **`./checker.py`** (notice the .py extension) in the `/render` directory. If you run it on the starter code, the program will print a table like the following, along with the results of our entire test set:

```
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2698           | (F)             | 0               |
| rand10k         | 2.7341           | (F)             | 0               |
| rand100k        | 26.1481          | (F)             | 0               |
| pattern         | 0.3591           | (F)             | 0               |
| snowsingle      | 16.1636          | (F)             | 0               |
| biglittle       | 14.9861          | (F)             | 0               |
| rand1M          | 188.0086         | (F)             | 0               |
| micro2M         | 355.9104         | (F)             | 0               |
--------------------------------------------------------------------------
|                                    | Total score:    | 0/72            |
--------------------------------------------------------------------------
```

Note: on some runs, you _may_ receive credit for some of these scenes, since the provided renderer's runtime is non-deterministic sometimes it might be correct. This doesn't change the fact that the current CUDA renderer is in general incorrect.

"Ref time" is the performance of our reference solution on your current machine (in the provided `render_ref` executable). "Your time" is the performance of your current CUDA renderer solution, where an `(F)` indicates an incorrect solution. Your grade will depend on the performance of your implementation compared to these reference implementations (see Grading Guidelines).

Along with your code, we would like you to hand in a clear, high-level description of how your implementation works as well as a brief description of how you arrived at this solution. Specifically address approaches you tried along the way, and how you went about determining how to optimize your code (For example, what measurements did you perform to guide your optimization efforts?).

Aspects of your work that you should mention in the write-up include:

1. Include both partners names and SUNet id's at the top of your write-up.
2. Replicate the score table generated for your solution and specify which machine you ran your code on.
3. Describe how you decomposed the problem and how you assigned work to CUDA thread blocks and threads (and maybe even warps).
4. Describe where synchronization occurs in your solution.
5. What, if any, steps did you take to reduce communication requirements (e.g., synchronization or main memory bandwidth requirements)?
6. Briefly describe how you arrived at your final solution. What other approaches did you try along the way. What was wrong with them?

### Grading Guidelines

- The write-up for the assignment is worth 7 points.
- Your implementation is worth 72 points. These are equally divided into 9 points per scene as follows:
  - 2 correctness points per scene.
  - 7 performance points per scene (only obtainable if the solution is correct). Your performance will be graded with respect to the performance of a provided benchmark reference renderer, T<sub>ref</sub>:
    - No performance points will be given for solutions having time (T) 10 times the magnitude of T<sub>ref</sub>.
    - Full performance points will be given for solutions within 20% of the optimized solution ( T <= 1.20 \* T<sub>ref</sub> )
    - For other values of T (for 1.20 T<sub>ref</sub> < T < 10 _ T<sub>ref</sub>), your performance score on a scale 1 to 7 will be calculated as: `7 _ T_ref / T`.
- Your implementation's performance on the class leaderboard is worth the final 6 points. Submission and grading details for the leaderboard will be detailed in a subsequent Ed post.

- Up to five points extra credit (instructor discretion) for solutions that achieve significantly greater performance than required. Your write up must clearly explain your approach thoroughly.
- Up to five points extra credit (instructor discretion) for a high-quality parallel CPU-only renderer implementation that achieves good utilization of all cores and SIMD vector units of the cores. Feel free to use any tools at your disposal (e.g., SIMD intrinsics, ISPC, pthreads). To receive credit you should analyze the performance of your GPU and CPU-based solutions and discuss the reasons for differences in implementation choices made.

So the total points for this project is as follows:

- part 1 (5 points)
- part 2 (10 points)
- part 3 write up (7 points)
- part 3 implementation (72 points)
- part 3 leaderboard (6 points)
- potential **extra** credit (up to 10 points)

## Assignment Tips and Hints

Below are a set of tips and hints compiled from previous years. Note that there are various ways to implement your renderer and not all hints may apply to your approach.

- There are two potential axes of parallelism in this assignment. One axis is _parallelism across pixels_ another is _parallelism across circles_ (provided the ordering requirement is respected for overlapping circles). Solutions will need to exploit both types of parallelism, potentially at different parts of the computation.
- The circle-intersects-box tests provided to you in `circleBoxTest.cu_inl` are your friend. You are encouraged to use these subroutines.
- The shared-memory prefix-sum operation provided in `exclusiveScan.cu_inl` may be valuable to you on this assignment (not all solutions may choose to use it). See the simple description of a prefix-sum [here](https://nvidia.github.io/cccl/thrust/api/function_group__prefixsums_1ga333bd4f34742dcf68d3ac5a0933f67db.html). We
  have provided an implementation of an exclusive prefix-sum on a **power-of-two-sized** arrays in shared memory. **The provided code does not work on non-power-of-two inputs and IT ALSO REQUIRES THAT THE NUMBER OF THREADS IN THE THREAD BLOCK BE THE SIZE OF THE ARRAY. PLEASE READ THE COMMENTS IN THE CODE.**
- Take a look at the `shadePixel` method that is being called. Notice how it is doing many global memory operations to update the color of a pixel. It might be wise to instead use a local accumulator in your `kernelRenderCircles` method. You can then perform the accumulation of a pixel value in a register, and once the final pixel value is accumulated you can then just perform a single write to global memory.
- You are allowed to use the [Thrust library](http://thrust.github.io/) in your implementation if you so choose. Thrust is not necessary to achieve the performance of the optimized CUDA reference implementations. There is one popular way of solving the problem that uses the shared memory prefix-sum implementation that we give you. There another popular way that uses the prefix-sum routines in the Thrust library. Both are valid solution strategies.
- Is there data reuse in the renderer? What can be done to exploit this reuse?
- How will you ensure atomicity of image update since there is no CUDA language primitive that performs the logic of the image update operation atomically? Constructing a lock out of global memory atomic operations is one solution, but keep in mind that even if your image update is atomic, the updates must be performed in the required order. **We suggest that you think about ensuring order in your parallel solution first, and only then consider the atomicity problem (if it still exists at all) in your solution.**
- For the tests which contain a larger number of circles - `rand1M` and `micro2M` - you should be careful about allocating temporary structures in global memory. If you allocate too much global memory, you will have used up all the memory on the device. If you are not checking the `cudaError_t` value that is returned from a call to `cudaMalloc`, then the program will still execute but you will not know that you ran out of device memory. Instead, you will fail the correctness check because you were not able to make your temporary structures. This is why we suggest you to use the CUDA API call wrapper below so you can wrap your `cudaMalloc` calls and produce an error when you run out of device memory.
- If you find yourself with free time, have fun making your own scenes!

### Catching CUDA Errors

By default, if you access an array out of bounds, allocate too much memory, or otherwise cause an error, CUDA won't normally inform you; instead it will just fail silently and return an error code. You can use the following macro (feel free to modify it) to wrap CUDA calls:

```
#define DEBUG

#ifdef DEBUG
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
#else
#define cudaCheckError(ans) ans
#endif
```

Note that you can undefine DEBUG to disable error checking once your code is correct for improved performance.

You can then wrap CUDA API calls to process their returned errors as such:

```
cudaCheckError( cudaMalloc(&a, size*sizeof(int)) );
```

Note that you can't wrap kernel launches directly. Instead, their errors will be caught on the next CUDA call you wrap:

```
kernel<<<1,1>>>(a); // suppose kernel causes an error!
cudaCheckError( cudaDeviceSynchronize() ); // error is printed on this line
```

All CUDA API functions, `cudaDeviceSynchronize`, `cudaMemcpy`, `cudaMemset`, and so on can be wrapped.

**IMPORTANT:** if a CUDA function error'd previously, but wasn't caught, that error will show up in the next error check, even if that wraps a different function. For example:

```
...
line 742: cudaMalloc(&a, -1); // executes, then continues
line 743: cudaCheckError(cudaMemcpy(a,b)); // prints "CUDA Error: out of memory at cudaRenderer.cu:743"
...
```

Therefore, while debugging, it's recommended that you wrap **all** CUDA API calls (at least in code that you wrote).

(Credit: adapted from [this Stack Overflow post](https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api))

