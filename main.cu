#include "gpu.h"


__global__ void render(vec3 *frameBuffer, int pixels_x, int pixels_y , vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
    //frameBuffer[0] = 0.2;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= pixels_x) || (j >= pixels_y)) return;

    int pixel_index = j*pixels_x + i;

    float u = float(i) / float(pixels_x);
    float v = float(j) / float(pixels_y);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);

    frameBuffer[pixel_index] = colour(r);
    //overload assignment
    //(frameBuffer[pixel_index]) = vec3(float(i) / pixels_x, float(j) / pixels_y, 0.2f);

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

void* allocateFb(vec3* d_fb) {
    int num_pixels = nx*ny;
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

     
   /* int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(float);

    checkCudaErrors(cudaMalloc((void**)&d_fb, fb_size));*/
    //d_fb = (float*)allocateFb(d_fb);


    render<<<blocks, threads>>>(d_fb, nx, ny,
                                vec3(-2.0, -1.0, -1.0),
                                vec3(4.0, 0.0, 0.0),
                                vec3(0.0, 2.0, 0.0),
                                vec3(0.0, 0.0, 0.0));

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    //checkCudaErrors(cudaMemcpy(h_fb,d_fb,fb_size, cudaMemcpyDeviceToHost));

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


__device__ vec3 colour(const ray &r) {
    
    vec3 unit_direction = unit_vector(r.direction());

    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t*vec3(0.5f, 0.7f, 1.0f);
}
