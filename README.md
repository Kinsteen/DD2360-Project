# DD2360 Final Project: Ray tracing in a weekend (but parallelized)

## Multi threaded version
Divide the image into smaller tiles (32x32 for example), and launch one thread per core to render a single tile. When a thread finishes, create a new one with another position. Repeat until image is done.

A class "Tile" will describe the result of the work. It would have:
- x and y position
- size_x and size_y (should be the same)
- sample size (maybe implement dynamic sampling?)
- array of computed pixels

A code in the main thread will then combine those arrays into a single pixel array, to avoid concurrent writes to a single array.
We could use mutex, but then the code will need to be rewritten to be ported to CUDA.

### Final implementation
I had problems with tiles of different sizes with the existing code base, so it will be only square tiles, which is great. The code generates more pixel than the image because of the tiling, so those pixels are cut from the result.

If we could do different tile sizes, it would maybe improve performance a bit.

In the implementation, every tile has a thread, and every thread in launched simultaneously. This may have some performance impact, and could be fixed by launching threads when other are finished. This however is a bit difficult, and I'm not sure if it is necessary for the CUDA porting to come.

The merging code works pretty well, but readability could be improved and performance should also be of concerns. I'm not sure if there's a more optimal way than just cycling through every pixel in a tile, but maybe.

## CUDA version
A lot of the code can be reused. Every function running on device should have `__device__` identifier, and most of the code works.

std::vectors and shared pointers do not exist in CUDA, so they need to be replaced with static arrays.

Random is also a problem as rand() is not available on device. Curand makes possible to generate random numbers on device, but it's a bit more complicated to use. In this implementation, we declare a global curand state on the device, and launch a kernel to initialize it on every kernel thread that will be launched.

### Results
- Single threaded, 5 samples, 500x281: 25 seconds
- Multi threaded, 5 samples, 500x281, 16 threads: 3.5 seconds (7 times faster)
- CUDA, 5 samples, 500x281: 1.31773 seconds (19 times faster than single thread CPU)
- CUDA (after all the optimizations), 5 samples, 500x281: 0.19s (131x faster than single thread CPU)
- CUDA, 30 samples, 1280x720, float: 38.6235 seconds
- CUDA, 30 samples, 1280x720, double: 84.5456 seconds
- Multi threaded, 30 samples, 1280x720, double: 161.626 seconds
- Multi threaded, 30 samples, 1280x720, float: 140.278 seconds

According to https://xmartlabs.github.io/cuda-calculator/, the optimal number of threads per block is 640, which is a tile of 32x20.
Resolution: 800x450, 32x32: 4.62s average
Resolution: 800x450, 20x32: 4.45s average (3.6% improvement)

From here, each run creates a line in the result file, so that we can make the program run in a loop and get results.

Removing recursion in ray_color makes code run way faster, around 40% faster!
- CUDA, 30 samples, 1280x720, float: 13.8342 seconds (64% improvement)
Seems like using 32x32 grid yields more perf?
Recursion or not doesn't change anything when running on CPU

Idea: use shared memory in each block for pixels instead of unified memory?

Performance is linear with items in scene: 30 objects -> 1.9s, 300 objects -> 17.7s.

Removing virtual functions (world/spheres) was 3.54x faster (there's still materials which uses virtual classes)

Registers per thread were 119, which limited 512 threads per block
25.0s with 16x16 blocks and no registers limit
maxrregcount=64 reduces the render time a lot (12.0s with 32x32, 10s with 16x16)
maxrregcount=40 is too low (16.3s)

Using relocatable device code is somehow faster (1.08x)

use_fast_math = 1.02x faster.