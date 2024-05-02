#include "vec3.h"

__host__ __device__ vec3::vec3() {

}

__host__ __device__ vec3::vec3(float e0,float e1,float e2) {
    this->e[0] = e0;
    this->e[1] = e1;
    this->e[2] = e2;
}

__host__ __device__ float vec3::x() const {
    return this->e[0];
}

__host__ __device__ float vec3::y() const {
    return this->e[1];

}

__host__ __device__ float vec3::z() const {
    return this->e[2];

}