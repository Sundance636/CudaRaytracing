#include "gpu.h"
#include "vec3.h"


__global__ void render(float *frameBuffer, int pixels_x, int pixels_y) {
    //frameBuffer[0] = 0.2;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= pixels_x) || (j >= pixels_y)) return;

    int pixel_index = j*pixels_x*3 + i*3;
    frameBuffer[pixel_index + 0] = float(i) / pixels_x;
    frameBuffer[pixel_index + 1] = float(j) / pixels_y;
    frameBuffer[pixel_index + 2] = 0.2;
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

void* allocateFb(float* d_fb) {
    int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(float);

    checkCudaErrors(cudaMalloc((void**)&d_fb, fb_size));
    //std::cout << d_fb << "\n";
    //checkCudaErrors(cudaMallocManaged((void **)&d_fb, fb_size));

    return d_fb;
}

void renderBuffer(float* d_fb,float* h_fb, int tx, int ty) {
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

void freeGPU(float* d_fb) {
    checkCudaErrors(cudaFree(d_fb));

}

void transferMem(float* h_fb,float* d_fb) {
    int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(float);
    std::cout << "Device frame buffer address: " << d_fb << "\n";
    std::cout << "Host frame buffer address: " << h_fb << "\n";
    std::cout << "FrameBuffer Size: " << fb_size << "\n";
    checkCudaErrors(cudaMemcpy(h_fb,d_fb,fb_size, cudaMemcpyDeviceToHost));



}