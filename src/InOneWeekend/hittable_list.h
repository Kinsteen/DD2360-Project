#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H
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

#include <memory>
#include <vector>

#include "hittable.h"
#include "rtweekend.h"
#include "sphere.h"

class world {
   public:
    __host__ __device__ world() {}
    __host__ __device__ world(sphere** objects, int size) {
        objects_array = objects;
        this->size = size;
    }

    __host__ __device__ bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec);

   public:
    sphere** objects_array;
    int size;
};

__host__ __device__ bool world::hit(const ray& r, float t_min, float t_max, hit_record& rec) {
    hit_record temp_rec;
    auto hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i < size; i++) {
        if (objects_array[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif
