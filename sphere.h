#ifndef __sphere_h__
#define __sphere_h__

#include "entity.h"

class sphere: public entity {

    private:
        vec3 center;
        float radius;


    public:
        __device__ sphere();
        __device__ sphere(vec3 center, float radius);
        __device__ virtual bool hit(const ray&, float, float, hit_record&) const;
        __device__ vec3 getCenter();
        __device__ float getRadius();
        __device__ void setCenter(vec3);
        __device__ void setRadius(float);



};


#endif