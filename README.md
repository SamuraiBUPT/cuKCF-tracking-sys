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

+ 初步实现MKCFup算法

+ 了解推理原理，做优化：

  + 重构代码，这个代码根本无法应用在工业上，必须重构为一份可移植的code
  + 可能的算法优化 - 能否使用CUDA：搭建起来之后，就尝试使用cuda算子进行操作，尝试进行性能优化
  + 推理方面优化



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