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
