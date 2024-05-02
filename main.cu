#include "gpu.h"


__global__ void render(vec3 *frameBuffer, int pixels_x, int pixels_y) {
    //frameBuffer[0] = 0.2;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= pixels_x) || (j >= pixels_y)) return;

    int pixel_index = j*pixels_x + i;

    //vec3 temp(float(i) / pixels_x, float(j) / pixels_y, 0.2f);

    //overload assignment
    (frameBuffer[pixel_index]) = vec3(float(i) / pixels_x, float(j) / pixels_y, 0.2f);

    /*
    frameBuffer[pixel_index + 0] = float(i) / pixels_x;
    frameBuffer[pixel_index + 1] = float(j) / pixels_y;
    frameBuffer[pixel_index + 2] = 0.2;
    */
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


    render<<<blocks, threads>>>(d_fb, nx, ny);

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