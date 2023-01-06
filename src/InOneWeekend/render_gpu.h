#ifndef RENDER_GPU_H
#define RENDER_GPU_H

__global__ void renderCuda(float4 *pixels, int image_width, int image_height, int samples, camera *cam, world **world_ptr, int max_depth) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < image_width && j < image_height) {
        float4 pixel_color = pixels[i + j * image_width];
        for (int s = 1; s <= samples; s++) {
            auto u = (i + random_float()) / (image_width - 1);
            auto v = (j + random_float()) / (image_height - 1);
            ray r = cam->get_ray(u, v);
            color c = ray_color_no_recur(r, **world_ptr, max_depth);
            pixel_color = make_float4(
                c.x() + pixel_color.x,
                c.y() + pixel_color.y,
                c.z() + pixel_color.z,
                1 + pixel_color.w);
            pixels[i + j * image_width] = pixel_color;
        }
    }
}

bool convertPixels(float4 *gpuPixels, sf::Uint8 *sfml_pixels, int image_width, int image_height, int samples_per_pixel) {
    bool finished = true;
    for (int x = 0; x < image_width; x++) {
        for (int y = 0; y < image_height; y++) {
            int i = x + y * image_width;
            float4 pixel = gpuPixels[x + (image_height - y - 1) * image_width];

            if ((int)pixel.w > 0) {
                auto r = pixel.x / pixel.w;
                auto g = pixel.y / pixel.w;
                auto b = pixel.z / pixel.w;
                float gamma = 2.6;
                r = pow(r, 1 / gamma);
                g = pow(g, 1 / gamma);
                b = pow(b, 1 / gamma);
                sfml_pixels[i * 4 + 0] = (int)clamp(r * 255, 0, 255);
                sfml_pixels[i * 4 + 1] = (int)clamp(g * 255, 0, 255);
                sfml_pixels[i * 4 + 2] = (int)clamp(b * 255, 0, 255);
                sfml_pixels[i * 4 + 3] = 255u;

                if (finished && (int)pixel.w < samples_per_pixel) {
                    finished = false;
                }
            } else {
                finished = false;
            }
        }
    }

    return finished;
}

#endif
