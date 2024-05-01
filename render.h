#include <iostream>
#include <SDL2/SDL.h>
#include <GL/gl.h>

#define bool int
#define false 0u
#define true 1u

void mainLoop(SDL_Window *,float*);
bool Input();
//void preDraw();
void Draw(SDL_Window*, float*);