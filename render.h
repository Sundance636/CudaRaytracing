#ifndef __render_h__
#define __render_h__

#include <iostream>
#include <SDL2/SDL.h>
#include <GL/gl.h>
#include "gpu.h"

#define bool int
#define false 0u
#define true 1u

void mainLoop(SDL_Window *,vec3 *);
bool Input();
//void preDraw();
float* vecToFb(vec3*);
void Draw(SDL_Window*, vec3 *);

#endif