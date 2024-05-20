#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

namespace py = pybind11;

// NOTICE: we just declare here, the implementation is in `fhog.cu`
void launch_cuda_kernel1(float *r, int *alfa, float *dx, float *dy, float *boundary_x, float *boundary_y, int height,
                         int width, int numChannels, int NUM_SECTOR);

void launch_cuda_kernel2(float *mapp, float *r, int *alfa, int *nearest, float *w, int k, int height, int width,
                         int sizeX, int sizeY, int p, int stringSize, int NUM_SECTOR);

void launch_cuda_kernel3(float *newData, float *partOfNorm, float *mappmap, int sizeX, int sizeY, int p, int xp, int pp,
                         int poN_size, int map_size);

void launch_cuda_kernel4(float *newData, float *mappmap, int p, int sizeX, int sizeY, int pp, int yp, int xp, float nx,
                         float ny, int map_size);

void launch_func1(torch::Tensor &r, torch::Tensor &alfa, torch::Tensor &dx, torch::Tensor &dy,
                  torch::Tensor &boundary_x, torch::Tensor &boundary_y, int height, int width, int numChannels,
                  int NUM_SECTOR);

void launch_func2(torch::Tensor &mapp, torch::Tensor &r, torch::Tensor &alfa, torch::Tensor &nearest, torch::Tensor &w,
                  int k, int height, int width, int sizeX, int sizeY, int p, int stringSize, int NUM_SECTOR);

void launch_func3(torch::Tensor &h_newData, torch::Tensor &h_partOfNorm, torch::Tensor &h_mappmap, int sizeX, int sizeY,
                  int p, int xp, int pp, int poN_size, int map_size);

void launch_func4(torch::Tensor &h_newData, torch::Tensor &h_mappmap, int p, int sizeX, int sizeY, int pp, int yp,
                  int xp, float nx, float ny, int map_size);