#include <curand.h>
#include <curand_kernel.h>
#include <sys/time.h>

#include <SFML/Graphics.hpp>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

#include "camera.h"
#include "color.h"
#include "cuda_utils.h"
#include "hittable_list.h"
#include "material.h"
#include "rtweekend.h"
#include "sphere.h"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__device__ __host__ double3 ray_color_cuda(const ray_cuda &r) {
    double3 unit_direction = normalize(r.direction());
    auto t = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - t) * make_double3(1.0, 1.0, 1.0) + t * make_double3(0.5, 0.7, 1.0);
}
 __device__ __host__ void write_color(double3 pixel_color) {
    auto r = pixel_color.x;
    auto g = pixel_color.y;
    auto b = pixel_color.z;

    // Replace NaN components with zero. See explanation in Ray Tracing: The Rest of Your Life.
    if (r != r) r = 0.0;
    if (g != g) g = 0.0;
    if (b != b) b = 0.0;

    // Divide the color by the number of samples and gamma-correct for gamma=2.0.
    // auto scale = 1.0 / samples_per_pixel;
    // r = sqrt(scale * r);
    // g = sqrt(scale * g);
    // b = sqrt(scale * b);

    // Write the translated [0,255] value of each color component.
    r = 256 * clamp(r, 0.0, 0.999);
    g = 256 * clamp(g, 0.0, 0.999);
    b = 256 * clamp(b, 0.0, 0.999);
    printf("%d %d %d\n", (int) r, (int) g, (int) b);
    // std::cout << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
    //     << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
    //     << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

__global__ void render(double3 *pixels, int image_width, int image_height, double3 origin, double3 horizontal, double3 vertical, double3 lower_left_corner) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    auto u = double(i) / (image_width - 1);
    auto v = double(j) / (image_height - 1);
    ray_cuda r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    double3 pixel_color = ray_color_cuda(r);
    if (i + j * image_width < 320*320) {
    pixels[i + j * image_width] = pixel_color;
    }
    //printf("(%d,%d) %d %d %d\n", i, j, pixel_color.x, pixel_color.y, pixel_color.z);
    //write_color(pixel_color);
}

int main(void) {
    std::cerr << "Test program"
              << "\n";

    // Image
    const auto aspect_ratio = 1;
    const int image_width = 320;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    // Camera

    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;

    auto origin = make_double3(0, 0, 0);
    auto horizontal = make_double3(viewport_width, 0, 0);
    auto vertical = make_double3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - make_double3(0, 0, focal_length);

    // Render

    std::cout << "P3\n"
              << image_width << " " << image_height << "\n255\n";

    size_t array_size = 10 * 10 * 32 * 32 * sizeof(double3);

    double3 *pixels;
    cudaMalloc(&pixels, array_size);

    double3 *pixels_host = (double3 *)malloc(array_size);
    memset(pixels_host, 0, array_size);

    double iStart = cpuSecond();
    render<<<dim3(20, 20), dim3(16, 16)>>>(pixels, image_width, image_height, origin, horizontal, vertical, lower_left_corner);

    int err = cudaDeviceSynchronize();

    std::cerr << "Elapsed GPU " << (cpuSecond() - iStart) << "\n";
    //std::cerr << "Err device sync " << err << "\n";

    err = cudaMemcpy(pixels_host, pixels, array_size, cudaMemcpyDeviceToHost);
    //std::cerr << "Err memcpy " << err << "\n";
    iStart = cpuSecond();
    for (int j = image_height-1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            auto u = double(i) / (image_width-1);
            auto v = double(j) / (image_height-1);
            ray_cuda r(origin, lower_left_corner + u*horizontal + v*vertical - origin);
            double3 pixel_color = ray_color_cuda(r);
            write_color(pixel_color);
        }
    }
    std::cerr << "Elapsed CPU " << (cpuSecond() - iStart) << "\n";

    for (long long i = 0; i < 320*320; i++) {
        write_color(pixels_host[i]);
    }

    std::cerr << "\nDone.\n";
}