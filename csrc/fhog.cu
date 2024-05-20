#include <stdexcept>
#include <string>

#define FLT_EPSILON 1e-7

// ============================================================
//                          CUDA Kernels
// ============================================================
__global__ void func1_cu(float *r, int *alfa, float *dx, float *dy, float *boundary_x, float *boundary_y, int height,
                         int width, int numChannels, int NUM_SECTOR) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= width && idx < (height - 1) * width && (idx % width) >= 1 && (idx % width) < (width - 1)) {
        int j = idx / width; // row
        int i = idx % width; // col

        int c = 0;
        float x = dx[j * width * numChannels + i * numChannels + c];
        float y = dy[j * width * numChannels + i * numChannels + c];
        r[idx] = sqrtf(x * x + y * y);

        for (int ch = 1; ch < numChannels; ++ch) {
            int ch_idx = j * width * numChannels + i * numChannels + ch;
            float tx = dx[ch_idx];
            float ty = dy[ch_idx];
            float magnitude = sqrtf(tx * tx + ty * ty);
            if (magnitude > r[idx]) {
                r[idx] = magnitude;
                c = ch;
                x = tx;
                y = ty;
            }
        }

        float mmax = boundary_x[0] * x + boundary_y[0] * y;
        int maxi = 0;

        for (int kk = 0; kk < NUM_SECTOR; ++kk) {
            float dotProd = boundary_x[kk] * x + boundary_y[kk] * y;
            if (dotProd > mmax) {
                mmax = dotProd;
                maxi = kk;
            } else if (-dotProd > mmax) {
                mmax = -dotProd;
                maxi = kk + NUM_SECTOR;
            }
        }

        alfa[idx * 2 + 0] = maxi % NUM_SECTOR;
        alfa[idx * 2 + 1] = maxi;
    }
}

__global__ void func2_cu(float *mapp, float *r, int *alfa, int *nearest, float *w, int k, int height, int width,
                         int sizeX, int sizeY, int p, int stringSize, int NUM_SECTOR) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeX * sizeY)
        return;

    int i = idx / sizeX;
    int j = idx % sizeX;

    for (int ii = 0; ii < k; ++ii) {
        for (int jj = 0; jj < k; ++jj) {
            int k_i = k * i + ii;
            int k_j = j * k + jj;

            if (k_i > 0 && k_i < height - 1 && k_j > 0 && k_j < width - 1) {
                float r_val = r[k_i * width + k_j];
                float w_ii_0 = w[ii * 2 + 0];
                float w_ii_1 = w[ii * 2 + 1];
                float w_jj_0 = w[jj * 2 + 0];
                float w_jj_1 = w[jj * 2 + 1];

                int base_idx = i * stringSize + j * p;
                int alfa_idx_0 = alfa[(k_i * width + k_j) * 2 + 0];
                int alfa_idx_1 = alfa[(k_i * width + k_j) * 2 + 1];

                atomicAdd(&mapp[base_idx + alfa_idx_0], r_val * w_ii_0 * w_jj_0);
                atomicAdd(&mapp[base_idx + alfa_idx_1 + NUM_SECTOR], r_val * w_ii_0 * w_jj_0);

                if (i + nearest[ii] >= 0 && i + nearest[ii] <= sizeY - 1) {
                    atomicAdd(&mapp[(i + nearest[ii]) * stringSize + j * p + alfa_idx_0], r_val * w_ii_1 * w_jj_0);
                    atomicAdd(&mapp[(i + nearest[ii]) * stringSize + j * p + alfa_idx_1 + NUM_SECTOR],
                              r_val * w_ii_1 * w_jj_0);
                }

                if (j + nearest[jj] >= 0 && j + nearest[jj] <= sizeX - 1) {
                    atomicAdd(&mapp[i * stringSize + (j + nearest[jj]) * p + alfa_idx_0], r_val * w_ii_0 * w_jj_1);
                    atomicAdd(&mapp[i * stringSize + (j + nearest[jj]) * p + alfa_idx_1 + NUM_SECTOR],
                              r_val * w_ii_0 * w_jj_1);
                }

                if (i + nearest[ii] >= 0 && i + nearest[ii] <= sizeY - 1 && j + nearest[jj] >= 0 &&
                    j + nearest[jj] <= sizeX - 1) {
                    atomicAdd(&mapp[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * p + alfa_idx_0],
                              r_val * w_ii_1 * w_jj_1);
                    atomicAdd(&mapp[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * p + alfa_idx_1 + NUM_SECTOR],
                              r_val * w_ii_1 * w_jj_1);
                }
            }
        }
    }
}

__global__ void func3_cu(float *newData, float *partOfNorm, float *mappmap, int sizeX, int sizeY, int p, int xp,
                         int pp) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // Offset by 1 to match the loop in the Python code
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1; // Offset by 1 to match the loop in the Python code

    if (i <= sizeY && j <= sizeX) {
        int pos1 = i * (sizeX + 2) * xp + j * xp;
        int pos2 = (i - 1) * sizeX * pp + (j - 1) * pp;

        float valOfNorm;

        // Compute first norm
        valOfNorm = sqrtf(partOfNorm[i * (sizeX + 2) + j] + partOfNorm[i * (sizeX + 2) + (j + 1)] +
                          partOfNorm[(i + 1) * (sizeX + 2) + j] + partOfNorm[(i + 1) * (sizeX + 2) + (j + 1)]) +
                    FLT_EPSILON;
        for (int k = 0; k < p; ++k) {
            newData[pos2 + k] = mappmap[pos1 + k] / valOfNorm;
        }
        for (int k = 0; k < 2 * p; ++k) {
            newData[pos2 + 4 * p + k] = mappmap[pos1 + p + k] / valOfNorm;
        }

        // Compute second norm
        valOfNorm = sqrtf(partOfNorm[i * (sizeX + 2) + j] + partOfNorm[i * (sizeX + 2) + (j + 1)] +
                          partOfNorm[(i - 1) * (sizeX + 2) + j] + partOfNorm[(i - 1) * (sizeX + 2) + (j + 1)]) +
                    FLT_EPSILON;
        for (int k = 0; k < p; ++k) {
            newData[pos2 + p + k] = mappmap[pos1 + k] / valOfNorm;
        }
        for (int k = 0; k < 2 * p; ++k) {
            newData[pos2 + 6 * p + k] = mappmap[pos1 + p + k] / valOfNorm;
        }

        // Compute third norm
        valOfNorm = sqrtf(partOfNorm[i * (sizeX + 2) + j] + partOfNorm[i * (sizeX + 2) + (j - 1)] +
                          partOfNorm[(i + 1) * (sizeX + 2) + j] + partOfNorm[(i + 1) * (sizeX + 2) + (j - 1)]) +
                    FLT_EPSILON;
        for (int k = 0; k < p; ++k) {
            newData[pos2 + 2 * p + k] = mappmap[pos1 + k] / valOfNorm;
        }
        for (int k = 0; k < 2 * p; ++k) {
            newData[pos2 + 8 * p + k] = mappmap[pos1 + p + k] / valOfNorm;
        }

        // Compute fourth norm
        valOfNorm = sqrtf(partOfNorm[i * (sizeX + 2) + j] + partOfNorm[i * (sizeX + 2) + (j - 1)] +
                          partOfNorm[(i - 1) * (sizeX + 2) + j] + partOfNorm[(i - 1) * (sizeX + 2) + (j - 1)]) +
                    FLT_EPSILON;
        for (int k = 0; k < p; ++k) {
            newData[pos2 + 3 * p + k] = mappmap[pos1 + k] / valOfNorm;
        }
        for (int k = 0; k < 2 * p; ++k) {
            newData[pos2 + 10 * p + k] = mappmap[pos1 + p + k] / valOfNorm;
        }
    }
}

__global__ void func4_cu(float *newData, float *mappmap, int p, int sizeX, int sizeY, int pp, int yp, int xp, float nx,
                         float ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < sizeY && j < sizeX) {
        int pos1 = (i * sizeX + j) * p;
        int pos2 = (i * sizeX + j) * pp;

        // 切片操作1
        for (int jj = 0; jj < 2 * xp; ++jj) {
            float sum = 0.0f;
            for (int k = 0; k < yp; ++k) {
                sum += mappmap[pos1 + k * 2 * xp + yp * xp + jj];
            }
            newData[pos2 + jj] = sum * ny;
        }

        // 切片操作2
        for (int jj = 0; jj < xp; ++jj) {
            float sum = 0.0f;
            for (int k = 0; k < yp; ++k) {
                sum += mappmap[pos1 + k * xp + jj];
            }
            newData[pos2 + 2 * xp + jj] = sum * ny;
        }

        // 切片操作3
        for (int ii = 0; ii < yp; ++ii) {
            float sum = 0.0f;
            for (int k = 0; k < 2 * xp; ++k) {
                sum += mappmap[pos1 + yp * xp + ii * 2 * xp + k];
            }
            newData[pos2 + 3 * xp + ii] = sum * nx;
        }
    }
}

// ============================================================
//                          cuda wrappers
// ============================================================

// cuda wrapper for kernel 1
void launch_cuda_kernel1(float *r, int *alfa, float *dx, float *dy, float *boundary_x, float *boundary_y, int height,
                         int width, int numChannels, int NUM_SECTOR) {

    // kernel launch
    int blockSize = 128;
    int numBlocks = (height * width + blockSize - 1) / blockSize;

    func1_cu<<<numBlocks, blockSize>>>(r, alfa, dx, dy, boundary_x, boundary_y, height, width, numChannels, NUM_SECTOR);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    cudaDeviceSynchronize();
}

// cuda wrapper for kernel 2
void launch_cuda_kernel2(float *mapp, float *r, int *alfa, int *nearest, float *w, int k, int height, int width,
                         int sizeX, int sizeY, int p, int stringSize, int NUM_SECTOR) {
    // prepare pointer for device memory

    int blockSize = 128;
    int numBlocks = (sizeX * sizeY + blockSize - 1) / blockSize;

    func2_cu<<<numBlocks, blockSize>>>(mapp, r, alfa, nearest, w, k, height, width, sizeX, sizeY, p, stringSize,
                                       NUM_SECTOR);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    cudaDeviceSynchronize();
}

// cuda wrapper for kernel 3
void launch_cuda_kernel3(float *newData, float *partOfNorm, float *mappmap, int sizeX, int sizeY, int p, int xp, int pp,
                         int poN_size, int map_size) {

    dim3 blockSize(16, 16);
    dim3 numBlocks((sizeX + blockSize.x - 1) / blockSize.x, (sizeY + blockSize.y - 1) / blockSize.y);
    func3_cu<<<numBlocks, blockSize>>>(newData, partOfNorm, mappmap, sizeX, sizeY, p, xp, pp);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    cudaDeviceSynchronize();
}

// cuda wrapper for kernel 4
void launch_cuda_kernel4(float *newData, float *mappmap, int p, int sizeX, int sizeY, int pp, int yp, int xp, float nx,
                         float ny, int map_size) {

    dim3 blockSize(16, 16);
    dim3 numBlocks((sizeX + blockSize.x - 1) / blockSize.x, (sizeY + blockSize.y - 1) / blockSize.y);
    func4_cu<<<numBlocks, blockSize>>>(newData, mappmap, p, sizeX, sizeY, pp, yp, xp, nx, ny);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    cudaDeviceSynchronize();
}