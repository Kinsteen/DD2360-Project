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

#include <curand.h>
#include <curand_kernel.h>
#include <sys/time.h>

#include <iostream>
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

color ray_color(const ray &r, const hittable &world, int depth) {
    hit_record rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return color(0, 0, 0);

    if (world.hit(r, 0.001, infinity, rec)) {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);
        return color(0, 0, 0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

hittable_list random_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}

typedef struct
{
    int r;
    int g;
    int b;
    int a;
} Pixel;

class Tile {
   private:
    float3 *pixel_array;
    std::thread thread;

   public:
    int x;
    int y;
    int tile_size;
    int samples;

    Tile(int x, int y, int tile_size, int samples);
    ~Tile();
    void render(int image_width, int image_height, camera cam, hittable_list world, int max_depth);
    void renderThread(int image_width, int image_height, camera cam, hittable_list world, int max_depth);
    void join();
    float3 *getPixels();
};

Tile::Tile(int x, int y, int tile_size, int samples) {
    this->x = x;
    this->y = y;
    this->tile_size = tile_size;
    this->samples = samples;
    this->pixel_array = (float3 *)malloc(tile_size * tile_size * sizeof(float3));
}

Tile::~Tile() {
    free(pixel_array);
}

void Tile::render(int image_width, int image_height, camera cam, hittable_list world, int max_depth) {
    // std::cerr << "Rendering tile (" << this->x << ", " << this->y << ")\n";
    //  for (int j = this->y + size_y - 1; j >= this->y; --j) {
    for (int j = this->y; j < this->y + tile_size; ++j) {
        for (int i = this->x; i < this->x + tile_size; ++i) {
            float3 pixel_color = make_float3(0, 0, 0);
            for (int s = 0; s < samples; ++s) {
                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                color c = ray_color(r, world, max_depth);
                float3 new_color = make_float3(c.x(), c.y(), c.z());
                pixel_color += new_color;
            }
            int tile_x = i - this->x;
            int tile_y = j - this->y;
            int array_idx = tile_x + tile_y * tile_size;
            this->pixel_array[array_idx] = pixel_color;
        }
    }
}

void Tile::renderThread(int image_width, int image_height, camera cam, hittable_list world, int max_depth) {
    std::thread thread(&Tile::render, this, image_width, image_height, cam, world, max_depth);
    this->thread = std::move(thread);
}

// __device__ inline double random_double_cuda(curandState *states) {
//     int id = threadIdx.x + blockDim.x * blockIdx.x;
//     // Returns a random real in [0,1).
//     return curand_uniform(&states[id]);
// }

// __device__ uint3 ray_color_cuda(const ray &r, const hittable &world, int depth) {
//     hit_record rec;

//     // If we've exceeded the ray bounce limit, no more light is gathered.
//     if (depth <= 0)
//         return make_uint3(0, 0, 0);

//     if (world.hit(r, 0.001, infinity, rec)) {
//         ray scattered;
//         uint3 attenuation;
//         if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
//             return attenuation * ray_color_cuda(scattered, world, depth - 1);
//         return make_uint3(0, 0, 0);
//     }

//     vec3 unit_direction = unit_vector(r.direction());
//     auto t = 0.5 * (unit_direction.y() + 1.0);
//     return (1.0 - t) * make_uint3(1.0, 1.0, 1.0) + t * make_uint3(0.5, 0.7, 1.0);
// }

// __global__ void renderCUDA(int x, int y, int tile_size, int samples, int image_width, int image_height, camera cam, hittable_list world, int max_depth, curandState *states, uint3 *pixel_array) {
//     int id = threadIdx.x + blockDim.x * blockIdx.x;
//     int seed = id;                          // different seed per thread
//     curand_init(seed, id, 0, &states[id]);  // 	Initialize CURAND

//     for (int j = y; j < y + tile_size; ++j) {
//         for (int i = x; i < x + tile_size; ++i) {
//             uint3 pixel_color = make_uint3(0, 0, 0);
//             for (int s = 0; s < samples; ++s) {
//                 auto u = (i + random_double_cuda(states)) / (image_width - 1);
//                 auto v = (j + random_double_cuda(states)) / (image_height - 1);
//                 ray r = cam.get_ray(u, v);
//                 pixel_color += ray_color_cuda(r, world, max_depth);
//             }
//             int tile_x = i - x;
//             int tile_y = j - y;
//             int array_idx = tile_x + tile_y * tile_size;
//             pixel_array[array_idx] = pixel_color;
//         }
//     }
// }

void Tile::join() {
    this->thread.join();
}

float3 *Tile::getPixels() {
    return this->pixel_array;
}

__host__ int main() {
    srand(time(NULL));

    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 300;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 15;
    const int max_depth = 50;
    const int tile_size = 16;

    std::cerr << "Image size: " << image_width << "x" << image_height << "\n";

    // World
    auto world = random_scene();

    // Camera
    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    // CUDA setup
    int NPB = 32;
    int TB = 32;
    curandState *dev_random;
    cudaMalloc((void **)&dev_random, NPB * TB * sizeof(curandState));

    // Render
    std::cout << "P3\n"
              << image_width << ' ' << image_height << "\n255\n";

    int total_tiles_x = (int)std::ceil((double)image_width / (double)tile_size);
    int total_tiles_x_pixels = total_tiles_x * tile_size;
    int total_tiles_y = (int)std::ceil((double)image_height / (double)tile_size);
    int total_tiles_y_pixels = total_tiles_y * tile_size;
    int nb_tiles = total_tiles_x * total_tiles_y;
    std::cerr << "total_tiles_x: " << total_tiles_x << "\n";
    std::cerr << "total_tiles_y: " << total_tiles_y << "\n";
    std::cerr << "Nb tile: " << nb_tiles << "\n";
    Tile *tiles[nb_tiles];

    auto n_threads = std::thread::hardware_concurrency();
    std::cerr << "Detected " << n_threads << " concurrent threads."
              << "\n";
    std::vector<std::thread> threads{n_threads};

    double iStart = cpuSecond();

    // Render each tile
    for (int i = nb_tiles - 1; i >= 0; i--) {
        Tile *tile = new Tile((i * tile_size) % total_tiles_x_pixels, (i / (total_tiles_x)) * tile_size, tile_size, samples_per_pixel);
        // tile->render(total_tiles_x_pixels, total_tiles_y_pixels, cam, world, max_depth);
        tile->renderThread(total_tiles_x_pixels, total_tiles_y_pixels, cam, world, max_depth);
        tiles[nb_tiles - 1 - i] = tile;
    }

    for (auto &t : tiles) {
        t->join();
    }

    double iElaps = cpuSecond() - iStart;

    std::cerr << "Done render in " << iElaps << " seconds\n";

    iStart = cpuSecond();

    float3 *full_pixel_array = (float3 *)malloc(image_width * image_height * sizeof(float3));

    for (int i = 0; i < nb_tiles; i++) {
        Tile *t = tiles[i];

        int length = tile_size * tile_size;
        // Index of final pixel array (size of final image)
        int start_idx = t->x + (image_height - tile_size - t->y) * image_width;

        float3 *pixels = t->getPixels();

        for (int p = 0; p < length; p++) {
            // bottom pixel line should go on top
            int final_idx = (start_idx + p % tile_size) + (length - 1 - p) / tile_size * image_width;
            // Ditch pixels that were calculated but are out of the image
            if (final_idx <= image_height * image_width && final_idx >= 0)
                full_pixel_array[final_idx] = pixels[p];
        }
    }

    // Read through the full pixel array to create the image
    // TODO: switch to png?
    for (int i = 0; i < image_width * image_height; i++) {
        write_color(std::cout, full_pixel_array[i], samples_per_pixel);
    }

    iElaps = cpuSecond() - iStart;

    std::cerr << "Done reconstructing image in " << (iElaps) << "s.\n";
}
