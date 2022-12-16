#ifndef RENDER_CPU_H
#define RENDER_CPU_H

std::mutex m;
int threadStarted = 0;

class Tile {
   private:
    int id;
    float4 *pixel_array;
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
    void renderThread(int image_width, int image_height, camera cam, hittable_list **world_ptr, int max_depth);
    float4 *getPixels();
};

Tile::Tile(int id, Tile **tiles, int x, int y, int tile_size, int samples) {
    this->id = id;
    this->tiles = tiles;
    this->x = x;
    this->y = y;
    this->tile_size = tile_size;
    this->samples = samples;
    this->pixel_array = (float4 *)malloc(tile_size * tile_size * sizeof(float4));
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

                float4 pixel_color = this->pixel_array[array_idx];
                auto u = (i + random_float()) / (image_width - 1);
                auto v = (j + random_float()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                color c = ray_color(r, world, max_depth);
                float4 new_color = make_float4(c.x(), c.y(), c.z(), s);
                pixel_color = make_float4(c.x() + pixel_color.x, c.y() + pixel_color.y, c.z() + pixel_color.z, s);
                this->pixel_array[array_idx] = pixel_color;
            }
        }
    }

    this->done = true;
    m.lock();
    threadStarted--;
    m.unlock();
}

void Tile::renderThread(int image_width, int image_height, camera cam, hittable_list **world_ptr, int max_depth) {
    hittable_list world = **world_ptr;
    std::thread thread(&Tile::render, this, image_width, image_height, cam, world, max_depth);
    this->thread = std::move(thread);
}

float4 *Tile::getPixels() {
    return this->pixel_array;
}

bool mergeTiles(sf::Uint8 *pixel_array, Tile **tiles, int nb_tiles, int tile_size, int image_height, int image_width, int samples) {
    bool tempFinished = true;
    for (int i = 0; i < nb_tiles; i++) {
        Tile *t = tiles[i];

        if (!t->started) {
            tempFinished = false;
            continue;  // If the tile hasn't started rendering it's doesn't make sense to try to merge it
        }

        int length = tile_size * tile_size;
        // Index of final pixel array (size of final image)
        int start_idx = t->x + (image_height - tile_size - t->y) * image_width;

        float4 *pixels = t->getPixels();

        for (int p = 0; p < length; p++) {
            // bottom pixel line should go on top
            int final_idx = (start_idx + p % tile_size) + (length - 1 - p) / tile_size * image_width;
            // Ditch pixels that were calculated but are out of the image or if they're black
            if (final_idx <= image_height * image_width && final_idx >= 0) {
                if (pixels[p].w > 0) {
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
                } else {
                    tempFinished = false;
                }
            }
        }
    }

    return tempFinished;
}

#endif
