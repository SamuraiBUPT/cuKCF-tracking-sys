#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// 声明 CUDA 内核函数 launcher，定义在.cu文件中
extern void launch_cuda_kernel(float *mapp, float *r, int *alfa, int *nearest, float *w, int k, int height, int width,
                               int sizeX, int sizeY, int p, int stringSize, int NUM_SECTOR);

void launch_func2(py::array_t<float> mapp, py::array_t<float> r, py::array_t<int> alfa, py::array_t<int> nearest,
                  py::array_t<float> w, int k, int height, int width, int sizeX, int sizeY, int p, int stringSize,
                  int NUM_SECTOR) {
    auto mapp_buf = mapp.request();
    auto r_buf = r.request();
    auto alfa_buf = alfa.request();
    auto nearest_buf = nearest.request();
    auto w_buf = w.request();

    float *mapp_ptr = static_cast<float *>(mapp_buf.ptr);
    float *r_ptr = static_cast<float *>(r_buf.ptr);
    int *alfa_ptr = static_cast<int *>(alfa_buf.ptr);
    int *nearest_ptr = static_cast<int *>(nearest_buf.ptr);
    float *w_ptr = static_cast<float *>(w_buf.ptr);

    launch_cuda_kernel(mapp_ptr, r_ptr, alfa_ptr, nearest_ptr, w_ptr, k, height, width, sizeX, sizeY, p, stringSize,
                       NUM_SECTOR);
}

PYBIND11_MODULE(cuKCF, m) {
    m.def("launch_func2", &launch_func2, "Launch the CUDA kernel", py::arg("mapp"), py::arg("r"), py::arg("alfa"),
          py::arg("nearest"), py::arg("w"), py::arg("k"), py::arg("height"), py::arg("width"), py::arg("sizeX"),
          py::arg("sizeY"), py::arg("p"), py::arg("stringSize"), py::arg("NUM_SECTOR"));
}
