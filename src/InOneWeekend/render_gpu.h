#ifndef RENDER_GPU_H
#define RENDER_GPU_H

__global__ void renderCuda(float4 *pixels, int image_width, int image_height, int samples, camera *cam, hittable_list **world_ptr, int max_depth) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < image_width && j < image_height) {
        float4 pixel_color = make_float4(0, 0, 0, 0);
        for (int s = 0; s < samples; s++) {
            auto u = (i + random_float()) / (image_width - 1);
            auto v = (j + random_float()) / (image_height - 1);
            ray r = cam->get_ray(u, v);
            color c = ray_color(r, **world_ptr, max_depth);
            float4 new_color = make_float4(c.x(), c.y(), c.z(), s);
            pixel_color = make_float4(
                c.x() + pixel_color.x,
                c.y() + pixel_color.y,
                c.z() + pixel_color.z,
                s + 1);
            pixels[i + j * image_width] = pixel_color;
        }
    }
}

bool convertPixels(float4 *gpuPixels, sf::Uint8 *sfml_pixels, int image_width, int image_height) {
    bool finished = true;
    for (int x = 0; x < image_width; x++) {
        for (int y = 0; y < image_height; y++) {
            int i = x + y * image_width;
            float4 pixel = gpuPixels[x + (image_height - y - 1) * image_width];

            if (pixel.w > 0) {
                auto scale = 1.0 / pixel.w;
                auto r = sqrt(scale * pixel.x);
                auto g = sqrt(scale * pixel.y);
                auto b = sqrt(scale * pixel.z);
                sfml_pixels[i * 4 + 0] = 256 * clamp(r, 0.0, 0.999);
                sfml_pixels[i * 4 + 1] = 256 * clamp(g, 0.0, 0.999);
                sfml_pixels[i * 4 + 2] = 256 * clamp(b, 0.0, 0.999);
                sfml_pixels[i * 4 + 3] = 255u;
            } else {
                finished = false;
            }
        }
    }

    return finished;
}

#endif
