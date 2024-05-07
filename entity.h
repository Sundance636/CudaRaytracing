#ifndef __entity_h__
#define __entity_h__

#include "ray.h"
#include "vec3.h"

struct hit_record {
    float t;//from parametrization
    vec3 point;
    vec3 normal;
};


class entity {


    public:
        //'abstract' function to be defined in subclasses
       __device__ virtual bool hit(const ray&, float, float, hit_record&) const;



};



#endif