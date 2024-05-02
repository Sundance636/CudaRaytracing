#include "render.h"
#include "gpu.h"
#include "vec3.h"

int main() {

    

    //Initialize SDL stuff
    SDL_Window *applicationWindow;
    SDL_GLContext GLContext;

    if((SDL_Init(SDL_INIT_VIDEO|SDL_INIT_AUDIO)==-1)) { 
        //printf("Could not initialize SDL: %s.\n", SDL_GetError());
        exit(-1);
    }

    applicationWindow = SDL_CreateWindow("Test App" , SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED, 640,480, SDL_WINDOW_OPENGL);

    if ( applicationWindow == NULL ) {
        //fprintf(stderr, "Couldn't set 640x480x8 video mode: %s\n",
        //                SDL_GetError());
        exit(1);
    }

    GLContext = SDL_GL_CreateContext(applicationWindow);

    if(GLContext == NULL) {
        //fprintf(stderr, "Couldn't create context: %s\n",
        //                SDL_GetError());
        exit(2);
    }

    // allocate space for Frame Buffer
    float *h_fb = nullptr;
    float *d_fb = nullptr;

    vec3 testVector(0.1f,0.2f,0.3f);// = vec3::vec3(0.1,0.2,0.3);



    int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(float);
    h_fb = (float*)malloc(fb_size);

    d_fb = (float*)allocateFb(d_fb);
    //std::cout << d_fb << "\n";


    int tx = 8;
    int ty = 8;

    // Render our buffer
    renderBuffer(d_fb, h_fb, tx, ty);
    transferMem(h_fb, d_fb);



    // Output FB as Image to stdout
    /*
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*3*nx + i*3;
            float r = h_fb[pixel_index + 0];
            float g = h_fb[pixel_index + 1];
            float b = h_fb[pixel_index + 2];
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    */
    

    mainLoop(applicationWindow,h_fb);

    //deallocates
    freeGPU(d_fb);
    free(h_fb);

    //cleaning nd quit routine
    SDL_DestroyWindow(applicationWindow);
    SDL_Quit();

    return 0;
}

void mainLoop(SDL_Window *window,float* fb) {

    bool gQuit = false;

    while(!gQuit) {

        gQuit = Input();
        Draw(window,fb);

        SDL_GL_SwapWindow(window);//refresh screen draw?
        

    }


}

void Draw(SDL_Window *window,float *fb) {
    glDrawPixels(640,480,GL_RGB,GL_FLOAT,fb);
    
}

bool Input() {
    SDL_Event e;

    while(SDL_PollEvent(&e) != 0) {
        if(e.type == SDL_QUIT) {
                printf("Quiting Window.\n");
                return true;
        }
    }
    return false;
}