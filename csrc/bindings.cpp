#include "libMkcfup.h" // 假设这是包含inference函数的头文件
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(mkcfup, m) {
    m.doc() = "Pybind11 interface for the MKCFup tracking operator";
    m.def("inference", &inference, "A function that performs tracking", py::arg("input_dir"), py::arg("output_dir"),
          py::arg("item"));
}
