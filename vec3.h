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
        

        __host__ __device__ float x() const;
        __host__ __device__ float y() const;
        __host__ __device__ float z() const;

        __host__ __device__ vec3 operator=(const vec3 &otherVector);
        //__host__ __device__ vec3 operator*(float);//define scaling vectors
        __host__ __device__ vec3 operator+(const vec3 &vector);//vector addition
        //__host__ __device__ vec3 operator*( const vec3 &vector, float scalar);
};



#endif