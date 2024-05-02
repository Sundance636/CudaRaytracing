#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#define nx 640
#define ny 480

void* allocateFb(float*);

void renderBuffer(float*,float*,int,int);
void freeGPU(float*);
void transferMem(float*,float*);