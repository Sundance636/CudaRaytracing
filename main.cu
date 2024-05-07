#include "gpu.h"


__global__ void render(vec3 *frameBuffer, int pixels_x, int pixels_y , vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= pixels_x) || (j >= pixels_y)) return;

    int pixel_index = j*pixels_x + i;

    float u = float(i) / float(pixels_x);//ratio representing the position of u
    float v = float(j) / float(pixels_y);//ratio representing the position of v


    //define rays as starting from the origin, and their direction
    //is dependant on the current UV coordinates (like a crt raster scan across display)
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);

    frameBuffer[pixel_index] = colour(r);
}


void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

void* allocateFb(vec3* d_fb, int width, int height) {
    int num_pixels = width*height;
    size_t fb_size = num_pixels*sizeof(vec3);

    checkCudaErrors(cudaMalloc((void**)&d_fb, fb_size));

    //std::cout << d_fb << "\n";
    //checkCudaErrors(cudaMallocManaged((void **)&d_fb, fb_size));

    return d_fb;
}

void renderBuffer(vec3* d_fb, int tx, int ty) {
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    //to a 4 by 3 aspect ratio window/ 3d space
    render<<<blocks, threads>>>(d_fb, nx, ny,
                                vec3(-4.0, -3.0, 0.0),//lowest left point of 3d space
                                vec3(8.0, 0.0, 0.0),//the width of space (pos and neg)
                                vec3(0.0, 6.0, 0.0),//height of the space
                                vec3(0.0, 0.0, 0.0));// where the origin is defined

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

}

void freeGPU(vec3* d_fb) {
    checkCudaErrors(cudaFree(d_fb));

}

void transferMem(vec3* h_fb,vec3* d_fb) {
    int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(float);
    std::cout << "Device frame buffer address: " << d_fb << "\n";
    std::cout << "Host frame buffer address: " << h_fb << "\n";
    std::cout << "FrameBuffer Size: " << fb_size << "\n";

    checkCudaErrors(cudaMemcpy(h_fb,d_fb,fb_size, cudaMemcpyDeviceToHost));



}

//determinies the colour of the pixel once the ray is cast
__device__ vec3 colour(const ray &r) {
    vec3 sphereCenter = vec3(0.0f,0.0f,-1.0f);

    float t = hit_sphere(sphereCenter, 0.5f, r);
    
    
    //distance of sphere from screen(z - component), t positive for infront
    if(t > 0.0 ) {
        //point on the sphere
        //so get the surface normal at that point

        //return vec3(1.0f, 0.0f, 0.0f);
        vec3 spherePoint = r.point_at_parameter(t);
        vec3 surfaceNormal = spherePoint - sphereCenter;
        surfaceNormal = unit_vector(surfaceNormal);

        return 0.5f * vec3(surfaceNormal.x() + 1.0f, surfaceNormal.y() + 1.0f, surfaceNormal.z()+1.0f);
    }
    
    vec3 unit_direction = unit_vector(r.direction());
    t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t*vec3(0.5f, 0.7f, 1.0f);
}


//checking if way hits the sphere
__device__ float hit_sphere(const vec3 &center, float Radius,ray r) {
    //from the textbook formulala

    // (A - C) , point/origin difference from center of sphere
    vec3 distance = r.origin() - center;

    // vector form t*t*dot(B,B) + 2*t*dot(A-C,A-C) + dot(C,C) - R*R = 0
    // to the form at^2 + bt + c

    // dot(B,B)
    float a = dot_product(r.direction(),r.direction());

    //2*dot(A-C,A-C)
    float b = 2.0f * dot_product( distance, distance);

    // dot(A-C,A-C) - R^2
    float c = dot_product(center, center) - Radius*Radius;

    //disriminant (b^2 - 4ac):
    // < 0 no solutions
    // == 0 one solution
    // > 1 two solutions
    float discriminant = b*b  - 4 *a*c;

    return (discriminant > 0);


    //bad
    if(discriminant < 0) {
        return -1.0f;
    }
    else {
        //solve for 't' using the quadratic formula ( - for the closest point)
        return ((-1.0f * b) - sqrt(discriminant)) / (2.0f*a);
    }
    

}