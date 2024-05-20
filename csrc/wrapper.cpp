#include "wrapper.h"

namespace py = pybind11;

/*
 * @param r: OUTPUT
 * @param alfa: OUTPUT
 */
void launch_func1(torch::Tensor &r, torch::Tensor &alfa, torch::Tensor &dx, torch::Tensor &dy,
                  torch::Tensor &boundary_x, torch::Tensor &boundary_y, int height, int width, int numChannels,
                  int NUM_SECTOR) {
    // get data
    float *r_ptr = r.data_ptr<float>();
    int *alfa_ptr = alfa.data_ptr<int>();

    float *dx_ptr = dx.data_ptr<float>();
    float *dy_ptr = dy.data_ptr<float>();
    float *boundary_x_ptr = boundary_x.data_ptr<float>();
    float *boundary_y_ptr = boundary_y.data_ptr<float>();

    // launch CUDA kernel
    launch_cuda_kernel1(r_ptr, alfa_ptr, dx_ptr, dy_ptr, boundary_x_ptr, boundary_y_ptr, height, width, numChannels,
                        NUM_SECTOR);
}

/*
 * @param mapp: OUTPUT
 *
 */
void launch_func2(torch::Tensor &mapp, torch::Tensor &r, torch::Tensor &alfa, torch::Tensor &nearest, torch::Tensor &w,
                  int k, int height, int width, int sizeX, int sizeY, int p, int stringSize, int NUM_SECTOR) {
    // get data
    float *mapp_ptr = mapp.data_ptr<float>();
    float *r_ptr = r.data_ptr<float>();
    int *alfa_ptr = alfa.data_ptr<int>();
    int *nearest_ptr = nearest.data_ptr<int>();
    float *w_ptr = w.data_ptr<float>();

    launch_cuda_kernel2(mapp_ptr, r_ptr, alfa_ptr, nearest_ptr, w_ptr, k, height, width, sizeX, sizeY, p, stringSize,
                        NUM_SECTOR);
}

void launch_func3(torch::Tensor &h_newData, torch::Tensor &h_partOfNorm, torch::Tensor &h_mappmap, int sizeX, int sizeY,
                  int p, int xp, int pp, int poN_size, int map_size) {
    // get data
    float *h_newData_ptr = h_newData.data_ptr<float>();
    float *h_partOfNorm_ptr = h_partOfNorm.data_ptr<float>();
    float *h_mappmap_ptr = h_mappmap.data_ptr<float>();

    launch_cuda_kernel3(h_newData_ptr, h_partOfNorm_ptr, h_mappmap_ptr, sizeX, sizeY, p, xp, pp, poN_size, map_size);
}

void launch_func4(torch::Tensor &h_newData, torch::Tensor &h_mappmap, int p, int sizeX, int sizeY, int pp, int yp,
                  int xp, float nx, float ny, int map_size) {
    // get data
    float *h_newData_ptr = h_newData.data_ptr<float>();
    float *h_mappmap_ptr = h_mappmap.data_ptr<float>();

    launch_cuda_kernel4(h_newData_ptr, h_mappmap_ptr, p, sizeX, sizeY, pp, yp, xp, nx, ny, map_size);
}

PYBIND11_MODULE(cuKCF, m) {
    m.def("launch_func1", &launch_func1, "Launch the CUDA kernel for func1");

    m.def("launch_func2", &launch_func2, "Launch the CUDA kernel for func2");

    m.def("launch_func3", &launch_func3, "Launch the CUDA kernel for func3");

    m.def("launch_func4", &launch_func4, "Launch the CUDA kernel for func4");
}
