#ifndef RAY_H
#define RAY_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "vec3.h"
#include "cuda_utils.h"


class ray {
    public:
        __host__ __device__ ray() {}
        __host__ __device__ ray(const point3& origin, const vec3& direction)
            : orig(origin), dir(direction), tm(0)
        {}

        __host__ __device__ ray(const point3& origin, const vec3& direction, float time)
            : orig(origin), dir(direction), tm(time)
        {}

        __host__ __device__ point3 origin() const  { return orig; }
        __host__ __device__ vec3 direction() const { return dir; }
        __host__ __device__ float time() const    { return tm; }

        __host__ __device__ point3 at(float t) const {
            return orig + t*dir;
        }

    public:
        point3 orig;
        vec3 dir;
        float tm;
};

class ray_cuda {
    public:
        ray_cuda() {}
        __host__ __device__ ray_cuda(const float3& origin, const float3& direction)
            : orig(origin), dir(direction), tm(0)
        {}

        __host__ __device__ ray_cuda(const float3& origin, const float3& direction, float time)
            : orig(origin), dir(direction), tm(time)
        {}

        __host__ __device__ float3 origin() const  { return orig; }
        __host__ __device__ float3 direction() const { return dir; }
        __host__ __device__ float time() const    { return tm; }

        __host__ __device__ float3 at(float t) const {
            return orig + t*dir;
        }

    public:
        float3 orig;
        float3 dir;
        float tm;
};

#endif
