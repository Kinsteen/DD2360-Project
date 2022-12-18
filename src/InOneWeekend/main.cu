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

#include <SFML/Graphics.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include "camera.h"
#include "color.h"
#include "cuda_utils.h"
#include "hittable_list.h"
#include "material.h"
#include "render.h"
#include "render_cpu.h"
#include "render_gpu.h"
#include "rtweekend.h"
#include "sphere.h"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__host__ __device__ void random_scene(hittable_list **world, hittable **objects_array, int image_width, int image_height) {
#ifdef __CUDA_ARCH__
    dev_image_height = image_height;
    dev_image_width = image_width;
#endif

    int idx = 0;

    auto ground_material = new lambertian(color(0.5, 0.5, 0.5));
    objects_array[idx++] = new sphere(point3(0, -1000, 0), 1000, ground_material);

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_float();
            point3 center(a + 0.9 * random_float(), 0.2, b + 0.9 * random_float());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material *sphere_material;

                if (choose_mat < 0.60) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = new lambertian(albedo);
                    objects_array[idx++] = new sphere(center, 0.2, sphere_material);
                } else if (choose_mat < 0.80) {
                    float light_power = 4;
                    sphere_material = new diffuse_light(color::random() * light_power);
                    objects_array[idx++] = new sphere(center, 0.2, sphere_material);
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_float(0, 0.5);
                    sphere_material = new metal(albedo, fuzz);
                    objects_array[idx++] = new sphere(center, 0.2, sphere_material);
                } else {
                    // glass
                    sphere_material = new dielectric(1.5);
                    objects_array[idx++] = new sphere(center, 0.2, sphere_material);
                }
            }
        }
    }

    auto material1 = new dielectric(1.5);
    objects_array[idx++] = new sphere(point3(0, 1, 0), 1.0, material1);

    auto material2 = new lambertian(color(0.4, 0.2, 0.1));
    objects_array[idx++] = new sphere(point3(-4, 1, 0), 1.0, material2);

    auto material3 = new metal(color(0.7, 0.6, 0.5), 0.0);
    objects_array[idx++] = new sphere(point3(4, 1, 0), 1.0, material3);

    // float light_power = 1;
    // auto material4 = new diffuse_light(color(light_power, light_power, light_power));
    // objects_array[idx++] = new sphere(point3(0, 1300, 0), 1000.0, material4);

    *world = new hittable_list(objects_array, idx);
}

// We can't generate the scene on host because of the use of virtual functions in the hittable objects.
// We need to do everything on the GPU
// This function takes a pointer to a world, and the GPU will create all objects in the scene
__global__ void random_scene_kernel(hittable_list **world, hittable **objects_array, int image_width, int image_height) {
    random_scene(world, objects_array, image_width, image_height);
}

__host__ int main(int argc, char *argv[]) {
    srand(time(NULL));

    if (argc == 1) {
        std::cerr << "Launch program with argument \"cpu\" or \"cuda\"" << std::endl;
        exit(1);
    }

    bool isCuda = strcmp(argv[1], "cuda") == 0;

    // Image
    const auto aspect_ratio = 16.f / 9.f;
    //const auto aspect_ratio = 1;
    const int image_width = 1280;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 1;  // 3 samples is the minimum to have a correct contrast / colors
    const int max_depth = 10; // 10 is virtually the same than 100+
    const int tile_size = 32;

    // Camera
    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    // Window setup
    sf::RenderWindow window(sf::VideoMode(image_width, image_height), "Ray Tracing with CUDA",
                            sf::Style::Titlebar | sf::Style::Close);
    auto desktop = sf::VideoMode::getDesktopMode();
    window.setPosition(sf::Vector2i(desktop.width / 2 - window.getSize().x / 2, desktop.height / 2 - window.getSize().y / 2 - 40));
    sf::Texture tex;
    sf::Sprite sprite;

    if (!tex.create(image_width, image_height)) {
        std::cerr << "Couldn't create texture!" << std::endl;
        return 1;
    }

    tex.setSmooth(false);

    sprite.setTexture(tex);

    double iStart = cpuSecond();

    float4 *pixels;

    int total_tiles_x = (int)std::ceil((float)image_width / (float)tile_size);
    int total_tiles_x_pixels = total_tiles_x * tile_size;
    int total_tiles_y = (int)std::ceil((float)image_height / (float)tile_size);
    int total_tiles_y_pixels = total_tiles_y * tile_size;
    int nb_tiles = total_tiles_x * total_tiles_y;
    Tile **tiles = (Tile **)malloc((nb_tiles + 1) * sizeof(Tile *));
    tiles[nb_tiles] = NULL;  // Avoid segfault on render function

    // According to occupation calculator, 640 TPB is the optimal number.
    int grid_height = 32;
    int grid_width = 32;

    if (isCuda) {
        // Increase the stack size
        // Can be removed if we remove recursions in kernels
        size_t stackSize;
        cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
        std::cerr << "GPU Stack size: " << stackSize << std::endl;
        //cudaDeviceSetLimit(cudaLimitStackSize, 65536);
        cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
        std::cerr << "GPU Stack size: " << stackSize << std::endl;

        int grid_x = ceil(image_width / (float)grid_width) + 1;
        int grid_y = ceil(image_height / (float)grid_height) + 1;

        size_t array_size = grid_x * grid_y * grid_width * grid_height * sizeof(float4);

        printf("grid_x %d / grid_y %d\n", grid_x, grid_y);

        cudaMallocManaged(&pixels, array_size);

        camera *dev_cam;
        cudaMalloc(&dev_cam, sizeof(camera));
        cudaMemcpy(dev_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice);

        curandState *curandStates = NULL;
        cudaMalloc(&curandStates, grid_x * grid_y * grid_width * grid_height * sizeof(curandState));
        init_rand<<<1, 1>>>(curandStates);
        init_random_cuda<<<dim3(grid_x, grid_y), dim3(grid_width, grid_height)>>>(image_width, image_height);

        printf("Init random synchronize: %d\n", cudaDeviceSynchronize());

        hittable_list **world;
        cudaMallocManaged(&world, sizeof(hittable_list *));

        hittable **objects_array;
        cudaMallocManaged(&objects_array, 1000 * sizeof(hittable *));

        random_scene_kernel<<<1, 1>>>(world, objects_array, image_width, image_height);

        printf("Random scene synchronize: %d\n", cudaDeviceSynchronize());
        renderCuda<<<dim3(grid_x, grid_y), dim3(grid_width, grid_height)>>>(pixels, image_width, image_height, samples_per_pixel, dev_cam, world, max_depth);
    } else {
        // Render
        std::cerr << "Image size: " << image_width << "x" << image_height << "\n";
        std::cerr << "total_tiles_x: " << total_tiles_x << "\n";
        std::cerr << "total_tiles_y: " << total_tiles_y << "\n";
        std::cerr << "Nb tile: " << nb_tiles << "\n";
        std::cerr << "Sample size: " << samples_per_pixel << "\n";

        // Render each tile
        hittable_list **world_cpu = (hittable_list **)malloc(sizeof(hittable_list *));
        hittable **objects_array_cpu = (hittable **)malloc(5000 * sizeof(hittable *));
        random_scene(world_cpu, objects_array_cpu, image_width, image_height);
        m.lock();
        for (int i = nb_tiles - 1; i >= 0; i--) {
            int pixel_x = (i * tile_size) % total_tiles_x_pixels;
            int pixel_y = (i / (total_tiles_x)) * tile_size;
            Tile *tile = new Tile(nb_tiles - 1 - i, tiles, pixel_x, pixel_y, tile_size, samples_per_pixel);
            tiles[nb_tiles - 1 - i] = tile;
            tile->renderThread(total_tiles_x_pixels, total_tiles_y_pixels, cam, world_cpu, max_depth);
        }
        m.unlock();
    }

    sf::Uint8 *pixel_array_sfml = (sf::Uint8 *)malloc(image_width * image_height * 4 * sizeof(sf::Uint8));
    memset(pixel_array_sfml, 0, image_width * image_height * 4 * sizeof(sf::Uint8));
    tex.update(pixel_array_sfml);

    bool renderFinished = false;

    while (window.isOpen()) {
        sf::Event event;

        if (!renderFinished) {
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed) window.close();
            }

            bool tempFinished;
            if (isCuda) {
                tempFinished = convertPixels(pixels, pixel_array_sfml, image_width, image_height);
            } else {
                tempFinished = mergeTiles(pixel_array_sfml, tiles, nb_tiles, tile_size, image_height, image_width, samples_per_pixel);
            }

            tex.update(pixel_array_sfml);

            window.clear();
            window.draw(sprite);
            window.display();

            if (tempFinished) {
                double iElaps = cpuSecond() - iStart;

                std::cerr << "Done render in " << iElaps << " seconds\n";

                std::ofstream outfile;
                outfile.open("result.txt", std::ios_base::app);  // append instead of overwrite
                outfile << iElaps << ";" << image_width << ";" << image_height << ";" << grid_width << ";" << grid_height << ";" << samples_per_pixel << std::endl;

                // Merge one last time to be sure that the image is complete
                if (!isCuda) mergeTiles(pixel_array_sfml, tiles, nb_tiles, tile_size, image_height, image_width, samples_per_pixel);
                tex.update(pixel_array_sfml);

                window.clear();
                window.draw(sprite);
                window.display();

                tex.copyToImage().saveToFile("render.png");

                renderFinished = true;
                // exit(0);
            }

            // Sleep to not update too often
            // 10% performance hit with 720p 5 samples on CPU
            // 6% performance hit with 1600p on GPU
            sf::sleep(sf::milliseconds(100));
        } else {
            window.waitEvent(event);
            if (event.type == sf::Event::Closed) window.close();
        }
    }
}
