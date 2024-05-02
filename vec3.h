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
};