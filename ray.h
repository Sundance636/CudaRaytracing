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


#endif