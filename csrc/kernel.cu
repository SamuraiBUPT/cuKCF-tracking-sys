#include <string>
#include <stdexcept>

extern "C" __global__ void func2_cu(float *mapp, float *r, int *alfa, int *nearest, float *w, int k, int height, int width, int sizeX, int sizeY, int p, int stringSize, int NUM_SECTOR) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeX * sizeY) return;

    int i = idx / sizeX;
    int j = idx % sizeX;

    for (int ii = 0; ii < k; ++ii) {
        for (int jj = 0; jj < k; ++jj) {
            if ((i * k + ii > 0) && (i * k + ii < height - 1) && (j * k + jj > 0) && (j * k + jj < width - 1)) {
                int index = i * stringSize + j * p + alfa[(k * i + ii) * width + (j * k + jj)];
                mapp[index] += r[(k * i + ii) * width + (j * k + jj)] * w[ii * k + jj];
                // 省略其他部分以保持简洁
            }
        }
    }
}

// 包装函数，用于从 C++ 调用 CUDA 内核函数
extern "C" void launch_cuda_kernel(float *mapp, float *r, int *alfa, int *nearest, float *w, int k, int height, int width, int sizeX, int sizeY, int p, int stringSize, int NUM_SECTOR) {
    int blockSize = 256;
    int numBlocks = (sizeX * sizeY + blockSize - 1) / blockSize;

    func2_cu<<<numBlocks, blockSize>>>(mapp, r, alfa, nearest, w, k, height, width, sizeX, sizeY, p, stringSize, NUM_SECTOR);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    cudaDeviceSynchronize();
}
