#ifndef __gpu_h__
#define __gpu_h__

#include <iostream>
#include "vec3.h"
#include "ray.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#define nx 640
#define ny 480

void* allocateFb(vec3*, int, int);

void renderBuffer(vec3*,int,int);
void freeGPU(vec3*);
void transferMem(vec3*,vec3*);

__device__ vec3 colour(const ray&);
__device__ bool hit_sphere(const vec3&, float, ray);

#endif