#ifndef RENDER_H
#define RENDER_H

// Common Render functions
__host__ __device__ color ray_color(const ray &r, hittable &world, int depth) {
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

__host__ __device__ color ray_color_no_recur(const ray &r, hittable &world, int depth) {
    hit_record rec;

    ray current_ray = r;
    color current_attenuation = color(1.0, 1.0, 1.0);

    for (int i = depth; i >= 0; i--) {
        if (world.hit(current_ray, 0.001, infinity, rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat_ptr->scatter(current_ray, rec, attenuation, scattered)) {
                current_ray = scattered;
                current_attenuation = attenuation * current_attenuation;
            } else {
                return color(0, 0, 0);
            }
        } else {
            vec3 unit_direction = unit_vector(current_ray.direction());
            auto t = 0.5 * (unit_direction.y() + 1.0);
            color c = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
            return c * current_attenuation;
        }
    }

    // execeeded recursion
    return color(0, 0, 0);
}

#endif
