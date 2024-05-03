#include "render.h"

int main() {

    

    //Initialize SDL stuff
    SDL_Window *applicationWindow;
    SDL_GLContext GLContext;

    if((SDL_Init(SDL_INIT_VIDEO|SDL_INIT_AUDIO)==-1)) { 
        printf("Could not initialize SDL: %s.\n", SDL_GetError());
        exit(-1);
    }

    applicationWindow = SDL_CreateWindow("Raytracer Engine" , SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED, 640,480, SDL_WINDOW_OPENGL);

    if ( applicationWindow == NULL ) {
        fprintf(stderr, "Couldn't set 640x480x8 video mode: %s\n",
                        SDL_GetError());
        exit(1);
    }

    GLContext = SDL_GL_CreateContext(applicationWindow);

    if(GLContext == NULL) {
        fprintf(stderr, "Couldn't create context: %s\n",
                        SDL_GetError());
        exit(2);
    }

    // allocate space for Frame Buffer
    vec3 *h_fb = nullptr;
    vec3 *d_fb = nullptr;


    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);
    int tx = 8;
    int ty = 8;

    h_fb = (vec3*)malloc(fb_size);
    d_fb = (vec3*)allocateFb(d_fb, nx, ny);


    // Render our buffer
    renderBuffer(d_fb, tx, ty);
    transferMem(h_fb, d_fb);//transfer mem from device to host


    mainLoop(applicationWindow,h_fb);

    //deallocates
    freeGPU(d_fb);
    free(h_fb);

    //cleaning nd quit routine
    SDL_DestroyWindow(applicationWindow);
    SDL_Quit();

    return 0;
}

void mainLoop(SDL_Window *window,vec3 * fb) {

    bool gQuit = false;

    
    while(!gQuit) {

        gQuit = Input();
        Draw(window,fb);

        SDL_GL_SwapWindow(window);//refresh screen draw?
        

    }


}

void Draw(SDL_Window *window,vec3 *fb) {
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

float* vecToFb(vec3* h_fb) {
    int num_pixels = nx*ny;
    float* newbuffer = (float*)malloc(3*num_pixels*sizeof(float));
    int j = 0;

    for(int i = 0; i < num_pixels*3; i+=3) {
        newbuffer[i] = (h_fb[j]).x();
        newbuffer[i+1] = (h_fb[j]).y();
        newbuffer[i+2] = (h_fb[j]).z();
        j++;
    }

    return newbuffer;
}