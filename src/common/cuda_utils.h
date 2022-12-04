#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

__device__ curandState* dev_states;
__device__ int dev_image_width;
__device__ int dev_image_height;

inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline __host__ __device__ bool operator==(float3 &a, float b)
{
    return a.x == b && a.y == b && a.z == b;
}

inline __host__ __device__ bool operator==(float4 &a, float b)
{
    return a.x == b && a.y == b && a.z == b;
}

inline __host__ __device__ float3 operator*(const float t, const float3 &a)
{
    return make_float3(a.x * t, a.y * t, a.z * t);
}

inline __host__ __device__ float3 operator*(const float3 &a, float t)
{
    return t * a;
}

inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

#endif