CC=g++
CFLAGS=-Wall
LDFLAGS=-lSDL2 -ldl
LDLIBS=/opt/cuda/lib/ -lcudart -lGL
NVCC=nvcc


default:
# Compile CUDA source file
	nvcc -c main.cu -o main.o

# Compile C++ source files
# 	g++ -Wall render.cpp -c -lSDL2 -ldl -o render.o
	$(CC) $(CFLAGS) render.cpp -c $(LDFLAGS) -o render.o

# Link object files
#	g++ main.o render.o -o cudaT -lSDL2 -ldl -L/opt/cuda/lib/ -lcudart -lGL
	$(CC) main.o render.o -o cudaT $(LDFLAGS) -L$(LDLIBS)

	./cudaT

cuda:
	nvcc main.cu -o cudaT
	nvprof ./cudaT > img.ppm
	loupe ./img.ppm

cuda2:
# Compile CUDA source file
	nvcc -c main.cu -o main.o

# Compile C++ source files
	g++ -Wall render.cpp -c -lSDL2 -ldl -o render.o

# Link object files
	g++ main.o render.o -o cudaT -lSDL2 -ldl -L/opt/cuda/lib/ -lcudart -lGL
# profiling
	nvprof ./cudaT > img.ppm

# launch initiallyy rendered image in loupe
#	loupe ./img.ppm

install:
# do nothing for now

clean:
	rm ./*.o