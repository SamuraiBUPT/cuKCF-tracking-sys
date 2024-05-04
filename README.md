# 目标跟踪系统

*小组成员仅按姓氏首字母进行排序：*

+ 董亚竺 2021213628
+ 黄敬元 2021213640
+ 周睿杰 2021213611



# Introduction

**目标跟踪系统**是一个基于Kernelized Correlation Filters以及相关衍生算法的、实时跟踪目标的综合推理系统。我们将这一份课题解读为涵盖：

+ 提供追踪制定目标的功能
+ 提供推理系统UI界面
+ 内部的推理优化（Inference Speed optimizing, http requests processing）



# Roadmap

+ [x] 初步实现MKCFup算法 

+ [ ] 了解推理原理，做优化：
  + [x] 重构代码，这个代码根本无法应用在工业上，必须重构为一份可移植的code
  + [x] 可能的算法优化 - 能否使用CUDA：搭建起来之后，就尝试使用cuda算子进行操作，尝试进行性能优化
  + [ ] 推理方面优化



# Quick Start

__1. 安装opencv__

首先您需要安装opencv C++ 3.4的库

```bash
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.4

mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)  # 使用所有可用核心编译
make install
```

注意：如果使用的是opencv4以上的版本，可能会遇到编译问题。



__2. 安装fftw3__

```bash
apt-get install libfftw3-dev
```



__3. 安装fastapi, jinjia2等Python包__

```
pip install fastapi[all]
```



__4. 编译运行我们制作的Python binding__

```bash
git clone https://github.com/SamuraiBUPT/MKCFup-tracking-sys.git
cd MKCFup-tracking-sys
pip install -e .
```

这会自动开始编译C++文件。



然后您可以直接使用我们提供的基于fastapi的server进行测试：

```bash
cd server
python3 server.py
```

在`localhost:8000`您可以进行交互与查看部署结果。





# Optimize

## 1. Analysis

要使用cuda优化，不能盲目着手，要找出来到底什么地方是整个算法的bottleneck，并且使用cuda计算还要考虑memcpy的时间，如果时间瓶颈都比较小的话，那就没必要进行cuda计算。（因为无法trade off）

```bash
./MKCFup /d_workspace/KCFs-tracking-sys/sequences/Biker /d_workspace/KCFs-tracking-sys/res Biker
gprof MKCFup gmon.out > analysis.txt
```



在profile结果里面，rank前三的函数占比远超其他函数：

```
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ms/call  ms/call  name    
 21.43      0.15     0.15     3352     0.04     0.04  hogChannels(float*, float const*, float const*, int, int, int, float, int)
 15.71      0.26     0.11      779     0.14     0.27  gradHist(float*, float*, float*, int, int, int, int, int, bool)
 14.29      0.36     0.10      717     0.14     0.22  gradMag(float*, float*, float*, int, int, int, bool)
```



## 2. `hogChannels`优化

```cpp
    hogChannels(H + nbo * 0, R1, N, hb, wb, nOrients * 2, clip, 1);
    hogChannels(H + nbo * 2, R2, N, hb, wb, nOrients * 1, clip, 1);
    hogChannels(H + nbo * 3, R1, N, hb, wb, nOrients * 2, clip, 2);
```

hogChannels的调用有分块化的趋势，可以使用cuda进行优化。



## 3. `gradHist`优化

gradHist的bottleneck在于它的内部loop:

```cpp
    O0 = (int *)alMalloc(h * sizeof(int), 16);
    M0 = (float *)alMalloc(h * sizeof(float), 16);
    O1 = (int *)alMalloc(h * sizeof(int), 16);
    M1 = (float *)alMalloc(h * sizeof(float), 16);
    // main loop
    for (x = 0; x < w0; x++) {
        // compute target orientation bins for entire column - very fast
        gradQuantize(O + x * h, M + x * h, O0, O1, M0, M1, nb, h0, sInv2, nOrients, full, softBin >= 0);

        if (softBin < 0 && softBin % 2 == 0) {
        } else if (softBin % 2 == 0 || bin == 1) {
        } else {}
    }
    alFree(O0);
    alFree(O1);
    alFree(M0);
    alFree(M1);
```

中间的部分就省略了，对于这个函数，采用指令集优化可能是比cuda优化更好的方式。（还是那个原因，考虑tradeoff）

SIMD优化成功。



## 4. `gradMag`优化

SIMD优化





# Reference

## Papers

With respect, we have read multiple awesome papers related to KCF algorithm, listed as follows:

+ KCF算法：[High-Speed Tracking with Kernelized Correlation Filters](https://arxiv.org/pdf/1404.7584) **2014**
+ 进阶：[High Speed Tracking With A Fourier Domain Kernelized Correlation Filter](https://arxiv.org/pdf/1811.03236v1) **2018**
+ 进阶：[High-speed Tracking with Multi-kernel Correlation Filters](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tang_High-Speed_Tracking_With_CVPR_2018_paper.pdf) **2018**



## Github Code

+ [KCF C++ Implementation](https://github.com/foolwood/KCF)
+ [MKCFup C++  Implementation](https://github.com/tominute/MKCFup) **Core**
+ [KCF using GPU](https://github.com/denismerigoux/GPU-tracking)



## Blogs

+ [KCF CSDN Blog](https://blog.csdn.net/EasonCcc/article/details/79658928)
+ [KCF Zhihu Notebook](https://zhuanlan.zhihu.com/p/33543297)