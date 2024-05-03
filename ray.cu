#include "ray.h"

__device__ ray::ray() {
    //this->A = vec3(0.1f, 0.1f, 0.1f);
    //this->B = vec3(0.1f, 0.1f, 0.2f);
}

__device__ ray::ray(const vec3& a, const vec3& b) { 
    this->A = a;
    this->B = b;
}

__device__ vec3 ray::origin() const {
    return this->A;
}

__device__ vec3 ray::direction() const {
    return this->B;
}

__device__ vec3 ray::point_at_parameter(float t) const {
    return (this->A) + t*(this->B);
}