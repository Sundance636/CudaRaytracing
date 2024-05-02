#ifndef __ray_h__
#define __ray_h__

#include "vec3.h"

class ray {

    private:
        vec3 A;
        vec3 B;

    public:
        __device__ ray();
        __device__ ray(const vec3&, const vec3&);
        __device__ vec3 origin() const;
        __device__ vec3 direction() const;
        __device__ vec3 point_at_parameter(float) const;

    

};

__host__ __device__ vec3 vec3::operator=(const vec3 &otherVector) {
    this->e[0] = otherVector.e[0];
    this->e[1] = otherVector.e[1];
    this->e[2] = otherVector.e[2];


    return *this;
}

/*

__host__ __device__ vec3 vec3::operator*(float scalar) {
    return vec3( scalar * this->e[0], scalar * this->e[1], scalar * this->e[2] );
}

*/

__host__ __device__ vec3 operator*(const vec3 &v, float t) {
    return vec3(t* v.x(), t*v.y(), t*v.z());
}

__host__ __device__ vec3 operator*(float t, const vec3 &v) {
    return vec3(t* v.x(), t*v.y(), t*v.z());
}

__host__ __device__ vec3 operator+(const vec3 &Vector1, const vec3 &Vector2) {
    return vec3( Vector1.x() + Vector2.x(), Vector1.y() + Vector2.y(), Vector1.z() + Vector2.z());
}

#endif