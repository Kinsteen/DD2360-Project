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

std::mutex m;
int threadStarted = 0;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__host__ __device__
    color
    ray_color(const ray &r, hittable &world, int depth) {
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

__host__
    hittable_list *
    random_scene() {
    hittable **objects_array;
    cudaMallocManaged(&objects_array, sizeof(hittable *) * 5000);
    int idx = 0;

    auto ground_material = new lambertian(color(0.5, 0.5, 0.5));
    objects_array[idx++] = new sphere(point3(0, -1000, 0), 1000, ground_material);

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material *sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = new lambertian(albedo);
                    objects_array[idx++] = new sphere(center, 0.2, sphere_material);
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
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

    std::cerr << "Pushed " << idx << " objects" << std::endl;

    hittable_list *world_array = new hittable_list(objects_array, idx);

    return world_array;
}

// We can't generate the scene on host because of the use of virtual functions in the hittable objects.
// We need to do everything on the GPU
// This function takes a pointer to a world, and the GPU will create all objects in the scene
__global__ void random_scene_kernel(hittable_list **world, hittable **objects_array) {
    // hittable *objects_array[500];
    // cudaMalloc(&objects_array, sizeof(hittable*) * 5000);
    //  this doesn't work for some reason???
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 && j == 0) {
        int idx = 0;

        auto ground_material = new lambertian(color(0.5, 0.5, 0.5));
        objects_array[idx++] = new sphere(point3(0, -1000, 0), 1000, ground_material);

        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = random_double();
                point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

                if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                    material *sphere_material;

                    if (choose_mat < 0.8) {
                        // diffuse
                        auto albedo = color::random() * color::random();
                        sphere_material = new lambertian(albedo);
                        objects_array[idx++] = new sphere(center, 0.2, sphere_material);
                    } else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = color::random(0.5, 1);
                        auto fuzz = random_double(0, 0.5);
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

        // std::cerr << "Pushed " << idx << " objects" << std::endl;

        *world = new hittable_list(objects_array, idx);
        printf("world ptr 1st obj %p\n", (*world)->objects_array[0]);
    }
}

class Tile {
   private:
    int id;
    double4 *pixel_array;
    std::thread thread;
    Tile **tiles;

   public:
    int x;
    int y;
    int tile_size;
    int samples;
    bool done = false;
    bool started = false;

    Tile(int id, Tile **tiles, int x, int y, int tile_size, int samples);
    ~Tile();
    void render(int image_width, int image_height, camera cam, hittable_list world, int max_depth);
    void renderThread(int image_width, int image_height, camera cam, hittable_list *world, int max_depth);
    double4 *getPixels();
};

Tile::Tile(int id, Tile **tiles, int x, int y, int tile_size, int samples) {
    this->id = id;
    this->tiles = tiles;
    this->x = x;
    this->y = y;
    this->tile_size = tile_size;
    this->samples = samples;
    this->pixel_array = (double4 *)malloc(tile_size * tile_size * sizeof(double4));
}

Tile::~Tile() {
    free(pixel_array);
}

void Tile::render(int image_width, int image_height, camera cam, hittable_list world, int max_depth) {
    m.lock();
    while (true) {
        if (threadStarted < 16 && ((tiles[id + 1] != NULL && tiles[id + 1]->started) || (x == 0 && y == 0))) {
            threadStarted++;
            m.unlock();
            break;
        } else {
            m.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    started = true;
    for (int s = 0; s < samples; ++s) {
        for (int j = this->y; j < this->y + tile_size; ++j) {
            for (int i = this->x; i < this->x + tile_size; ++i) {
                int tile_x = i - this->x;
                int tile_y = j - this->y;
                int array_idx = tile_x + tile_y * tile_size;

                double4 pixel_color = this->pixel_array[array_idx];
                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                color c = ray_color(r, world, max_depth);
                double4 new_color = make_double4(c.x(), c.y(), c.z(), s);
                pixel_color = make_double4(c.x() + pixel_color.x, c.y() + pixel_color.y, c.z() + pixel_color.z, s);
                this->pixel_array[array_idx] = pixel_color;
            }
        }
    }

    this->done = true;
    m.lock();
    threadStarted--;
    m.unlock();
}

void Tile::renderThread(int image_width, int image_height, camera cam, hittable_list *world, int max_depth) {
    hittable_list temp = *world;
    std::thread thread(&Tile::render, this, image_width, image_height, cam, temp, max_depth);
    this->thread = std::move(thread);
}

double4 *Tile::getPixels() {
    return this->pixel_array;
}

__global__ void renderCuda(double4 *pixels, int image_width, int image_height, int samples, camera *cam, hittable_list **world_ptr, int max_depth) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    // const int id = i + j * blockIdx.y * blockDim.y;

    hittable_list temp_world = **world_ptr;

    if (i < image_width && j < image_height) {
        double4 pixel_color = make_double4(0, 0, 0, 0);
        for (int s = 1; s <= samples; ++s) {
            auto u = (i + random_double()) / (image_width - 1);
            auto v = (j + random_double()) / (image_height - 1);
            ray r = cam->get_ray(u, v);
            color c = ray_color(r, temp_world, max_depth);
            double4 new_color = make_double4(c.x(), c.y(), c.z(), s);
            pixel_color = make_double4(
                c.x() + pixel_color.x,
                c.y() + pixel_color.y,
                c.z() + pixel_color.z,
                s);
            pixels[i + j * image_width] = pixel_color;
        }
    }
}

void mergeTiles(sf::Uint8 *pixel_array, Tile **tiles, int nb_tiles, int tile_size, int image_height, int image_width, int samples) {
    for (int i = 0; i < nb_tiles; i++) {
        Tile *t = tiles[i];

        if (!t->started) continue;  // If the tile hasn't started rendering it's doesn't make sense to try to merge it

        int length = tile_size * tile_size;
        // Index of final pixel array (size of final image)
        int start_idx = t->x + (image_height - tile_size - t->y) * image_width;

        double4 *pixels = t->getPixels();

        for (int p = 0; p < length; p++) {
            // bottom pixel line should go on top
            int final_idx = (start_idx + p % tile_size) + (length - 1 - p) / tile_size * image_width;
            // Ditch pixels that were calculated but are out of the image or if they're black
            if (final_idx <= image_height * image_width && final_idx >= 0 && pixels[p].w > 0) {
                auto r = pixels[p].x;
                auto g = pixels[p].y;
                auto b = pixels[p].z;

                // Divide the color by the number of samples and gamma-correct for gamma=2.0.
                auto scale = 1.0 / pixels[p].w;
                r = sqrt(scale * r);
                g = sqrt(scale * g);
                b = sqrt(scale * b);

                pixel_array[final_idx * 4 + 0] = 256 * clamp(r, 0.0, 0.999);
                pixel_array[final_idx * 4 + 1] = 256 * clamp(g, 0.0, 0.999);
                pixel_array[final_idx * 4 + 2] = 256 * clamp(b, 0.0, 0.999);
                pixel_array[final_idx * 4 + 3] = 255u;
            }
        }
    }
}

void convertPixels(double4 *gpuPixels, sf::Uint8 *sfml_pixels, int image_width, int image_height) {
    for (int x = 0; x < image_width; x++) {
        for (int y = 0; y < image_height; y++) {
            int i = x + y * image_width;
            double4 pixel = gpuPixels[x + (image_height - y - 1) * image_width];

            auto scale = 1.0 / pixel.w;
            //printf("%f\n", pixel.w);
            auto r = sqrt(scale * pixel.x);
            auto g = sqrt(scale * pixel.y);
            auto b = sqrt(scale * pixel.z);
            sfml_pixels[i * 4 + 0] = 256 * clamp(r, 0.0, 0.999);
            sfml_pixels[i * 4 + 1] = 256 * clamp(g, 0.0, 0.999);
            sfml_pixels[i * 4 + 2] = 256 * clamp(b, 0.0, 0.999);
            sfml_pixels[i * 4 + 3] = 255u;
        }
    }
}

__host__ int main() {
    srand(time(NULL));

    // Image
    // const auto aspect_ratio = 16.f / 9.f;
    const auto aspect_ratio = 1;
    const int image_width = 640;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 20;  // 3 samples is the minimum to have a correct contrast / colors
    const int max_depth = 10;
    const int tile_size = 32;

    // World
    // auto world = random_scene();

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
    window.setPosition(sf::Vector2i(desktop.width / 2 - window.getSize().x / 2, desktop.height / 2 - window.getSize().y / 2));
    sf::Texture tex;
    sf::Sprite sprite;

    if (!tex.create(image_width, image_height)) {
        std::cerr << "Couldn't create texture!" << std::endl;
        return 1;
    }

    tex.setSmooth(false);

    sprite.setTexture(tex);

    // Render
    int total_tiles_x = (int)std::ceil((double)image_width / (double)tile_size);
    int total_tiles_x_pixels = total_tiles_x * tile_size;
    int total_tiles_y = (int)std::ceil((double)image_height / (double)tile_size);
    int total_tiles_y_pixels = total_tiles_y * tile_size;
    int nb_tiles = total_tiles_x * total_tiles_y;
    std::cerr << "Image size: " << image_width << "x" << image_height << "\n";
    std::cerr << "total_tiles_x: " << total_tiles_x << "\n";
    std::cerr << "total_tiles_y: " << total_tiles_y << "\n";
    std::cerr << "Nb tile: " << nb_tiles << "\n";
    std::cerr << "Sample size: " << samples_per_pixel << "\n";
    Tile **tiles = (Tile **)malloc(nb_tiles * sizeof(Tile *));

    double iStart = cpuSecond();

    // Increase the stack size
    // Can be removed if we remove recursions in kernels
    size_t stackSize;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    std::cerr << "GPU Stack size: " << stackSize << std::endl;
    cudaDeviceSetLimit(cudaLimitStackSize, 65536);
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    std::cerr << "GPU Stack size: " << stackSize << std::endl;

    int grid_height = 32;
    int grid_width = grid_height;
    int grid_x = ceil(image_width / (double)grid_width) + 1;
    int grid_y = ceil(image_height / (double)grid_height) + 1;

    size_t array_size = grid_x * grid_y * grid_width * grid_height * sizeof(double4);

    printf("grid_x %d / grid_y %d\n", grid_x, grid_y);

    double4 *pixels;
    cudaMallocManaged(&pixels, array_size);

    camera *dev_cam;
    printf("cudaMalloc %d\n", cudaMalloc(&dev_cam, sizeof(cam)));
    printf("cudaMemcpy %d\n", cudaMemcpy(dev_cam, &cam, sizeof(cam), cudaMemcpyHostToDevice));

    curandState *curandStates = NULL;
    cudaMalloc(&curandStates, grid_x * grid_y * grid_width * grid_height * sizeof(curandState));
    init_random_cuda<<<1, 1>>>(curandStates, rand());

    printf("%d\n", cudaDeviceSynchronize());

    hittable_list **world;
    cudaMalloc(&world, sizeof(hittable_list *));

    hittable **objects_array;
    cudaMalloc(&objects_array, 1000 * sizeof(hittable *));

    random_scene_kernel<<<1, 1>>>(world, objects_array);

    printf("device sync %d\n", cudaDeviceSynchronize());
    // renderCuda<<<dim3(grid_x, grid_y), dim3(grid_width, grid_height)>>>(pixels, image_width, image_height, samples_per_pixel, dev_cam, dev_world, max_depth);
    renderCuda<<<dim3(grid_x, grid_y), dim3(grid_width, grid_height)>>>(pixels, image_width, image_height, samples_per_pixel, dev_cam, world, max_depth);

    printf("device sync %d\n", cudaDeviceSynchronize());

    // Render each tile
    // m.lock();
    // for (int i = nb_tiles - 1; i >= 0; i--) {
    //     int pixel_x = (i * tile_size) % total_tiles_x_pixels;
    //     int pixel_y = (i / (total_tiles_x)) * tile_size;
    //     Tile *tile = new Tile(nb_tiles - 1 - i, tiles, pixel_x, pixel_y, tile_size, samples_per_pixel);
    //     tiles[nb_tiles - 1 - i] = tile;
    //     tile->renderThread(total_tiles_x_pixels, total_tiles_y_pixels, cam, *world, max_depth);
    // }
    // m.unlock();

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

            // mergeTiles(pixel_array_sfml, tiles, nb_tiles, tile_size, image_height, image_width, samples_per_pixel);

            convertPixels(pixels, pixel_array_sfml, image_width, image_height);

            tex.update(pixel_array_sfml);

            window.clear();
            window.draw(sprite);
            window.display();

            // Sleep to not update too often
            // 10% performance hit with 720p 5 samples
            sf::sleep(sf::milliseconds(100));

            // bool tempFinished = true;
            // for (int i = 0; i < nb_tiles; i++) {
            //     Tile *t = tiles[i];
            //     if (!t->done) {
            //         tempFinished = false;
            //         break;
            //     }
            // }

            // if (tempFinished) {
            //     double iElaps = cpuSecond() - iStart;

            //     std::cerr << "Done render in " << iElaps << " seconds\n";

            //     // Merge one last time to be sure that the image is complete
            //     mergeTiles(pixel_array_sfml, tiles, nb_tiles, tile_size, image_height, image_width, samples_per_pixel);
            //     tex.update(pixel_array_sfml);

            //     window.clear();
            //     window.draw(sprite);
            //     window.display();

            //     tex.copyToImage().saveToFile("render.png");

            //     renderFinished = true;
            // }
        } else {
            window.waitEvent(event);
            if (event.type == sf::Event::Closed) window.close();
        }
    }
}
