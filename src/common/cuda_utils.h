#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

__device__ curandState* dev_states;

inline __host__ __device__ void operator+=(double3 &a, double3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline __host__ __device__ bool operator==(double3 &a, double b)
{
    return a.x == b && a.y == b && a.z == b;
}

inline __host__ __device__ bool operator==(double4 &a, double b)
{
    return a.x == b && a.y == b && a.z == b;
}

inline __host__ __device__ double3 operator*(const double t, const double3 &a)
{
    return make_double3(a.x * t, a.y * t, a.z * t);
}

inline __host__ __device__ double3 operator*(const double3 &a, double t)
{
    return t * a;
}

inline __host__ __device__ double3 operator+(const double3 &a, const double3 &b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ double3 normalize(double3 v)
{
    double invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

inline __host__ __device__ double3 operator/(double3 a, double b)
{
    return make_double3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

#endif