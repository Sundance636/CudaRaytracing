#ifndef __vec_h__
#define __vec_h__

#include <cuda.h>
#include <cuda_runtime.h>



class vec3 {
    private:
    float e[3];


    public:
        __host__ __device__ vec3();
        __host__ __device__ vec3(float,float,float);
        

        __host__ __device__ float x();
        __host__ __device__ float y();
        __host__ __device__ float z();

        __host__ __device__ vec3 operator=(const vec3 &otherVector);
};

#endif