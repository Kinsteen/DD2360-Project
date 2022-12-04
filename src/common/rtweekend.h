#ifndef RTWEEKEND_H
#define RTWEEKEND_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

#include "cuda_utils.h"

// Usings

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

// Utility Functions

__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

__host__ __device__ inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__global__ void init_rand(curandState *states) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Set the device pointer to the new pointer set
        dev_states = states;
    }
}

__global__ void init_random_cuda(int image_width, int image_height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= image_width) || (j >= image_height)) return;

    int pixel_index = j * image_width + i;
    curand_init(clock64()+pixel_index, 0, 0, &dev_states[pixel_index]);
}

__host__ __device__ inline float random_float() {
// Returns a random real in [0,1).
#ifdef __CUDA_ARCH__
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    int pixel_index = j * dev_image_width + i;
    return curand_uniform(&dev_states[pixel_index]);
#else
    return rand() / (RAND_MAX + 1.0);
#endif
}

__host__ __device__ inline float random_float(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

__host__ __device__ inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return static_cast<int>(random_float(min, max + 1));
}

// Common Headers

#include "ray.h"
#include "vec3.h"

#endif
