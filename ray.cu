#include "ray.h"

__device__ ray::ray() {
    //this->A = vec3(0.1f, 0.1f, 0.1f);
    //this->B = vec3(0.1f, 0.1f, 0.2f);
}

__device__ ray::ray(const vec3& a, const vec3& b) { 
    this->A = a;//trust in the copy constructor/compiler
    this->B = b;
}

//return the point in space representing the origin
__device__ vec3 ray::origin() const {
    return this->A;
}

//return the direction vector of this ray
__device__ vec3 ray::direction() const {
    return this->B;
}

//basically parameterizing the path the ray takes so -> p(t) (where B is direction)
__device__ vec3 ray::point_at_parameter(float t) const {
    return (this->A) + t*(this->B);
}