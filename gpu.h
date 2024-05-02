#include <iostream>
#include "vec3.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#define nx 640
#define ny 480

void* allocateFb(vec3*);

void renderBuffer(vec3*,int,int);
void freeGPU(vec3*);
void transferMem(vec3*,vec3*);