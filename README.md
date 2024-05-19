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
  + [ ] 可能的算法优化 - 能否使用CUDA、SIMD指令集
  + [ ] 推理方面优化
+ [ ] 重构一版Python版本的算法
+ [ ] 前后端的软件工程问题：有一个图像标注的过程转化



# Quick Start

> 强烈建议使用linux环境，如果使用windows环境，遇到CMake编译问题请自行调整^ ^



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



+ 先找是否是精度问题
+ 再考虑profile bottleneck
+ 再考虑cuda优化



__1.1 精度问题__

我认为计算精确度的问题，应该是精度的问题，不是其他的。

如果尝试使用更高的精度，也许效果会更好。

**结论**：并没有更好



__1.2 bottleneck分析__

![image-20240519001146457](D:\Code_Space\Cplusplus\cpp_projects\workspace\KCFs-tracking-sys\README\profile1.png)



__初步profile结果：__

+ NormalizeAndTruncate里面调用了**func3.**  - rank No.3
+ getFeatureMaps调用了：**func1, func2**  - rank No.4, No.7
+ PCAFeatureMaps调用了：**func4**  - rank No.10
+ 还有一个就是**gaussianCorrelation**  - rank No.5
  + 包括mulSpectrums
+ FFTD和dft其实是一伙的。（fftd函数内部调用的就是cv2.dft） - rank No.2







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
+ 有关图像的梯度直方图讲解：https://zhuanlan.zhihu.com/p/85829145