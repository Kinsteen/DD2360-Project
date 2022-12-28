#ifndef RENDER_H
#define RENDER_H

// Common Render functions
__host__ __device__ color ray_color_no_recur(const ray &r, world &m_world, int depth) {
    hit_record rec;

    ray current_ray = r;
    color current_attenuation = color(1.0, 1.0, 1.0);

    for (int i = depth; i >= 0; i--) {
        if (m_world.hit(current_ray, 0.001, infinity, rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat_ptr->scatter(current_ray, rec, attenuation, scattered)) {
                current_ray = scattered;
                current_attenuation = attenuation * current_attenuation;
            } else {
                current_attenuation = rec.mat_ptr->emitted() * current_attenuation;
                return rec.mat_ptr->emitted() * current_attenuation;
            }
        } else {
            vec3 unit_direction = unit_vector(current_ray.direction());
            auto t = 0.5 * (unit_direction.y() + 1.0);
            color c = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
            //return c * current_attenuation;
            return color(0);
        }
    }

    // execeeded recursion
    return color(0, 0, 0);
}

#endif
