#pragma once

#include "ComplexMat.h"
#include "cnfeat.hpp"
#include "fhog.hpp"
#include "omp.h"
#include "params.h"

#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

typedef struct _COMPLEX {
    float real;
    float img;
} COMPLEX;

struct model {
    complex_mat alphaf;
    float d[2];
};

float **allocfloat(float **mem, int h, int w);
COMPLEX **alloccomplex(COMPLEX **mem, int h, int w);
void FFT2(float **input, COMPLEX **output, int height, int width);
void IFFT2(COMPLEX **input, float **output, int height, int width);

complex_mat newfft2(const Mat &input);
complex_mat newfft2(const Mat &input, const Mat &cos_window);
Mat newifft2(const complex_mat &input);
Mat newifft2_for_y(const complex_mat &input);
Mat cosine_window_function(int dim1, int dim2);
Mat scale_window_function(int dim1);
Mat circshift(const Mat &patch, int x_rot, int y_rot);
Mat gaussian_shaped_labels(double sigma, int dim1, int dim2);
Mat get_subwindow(const Mat &input, int cx, int cy, int width, int height, float currentScaleFactor);
vector<Mat> average_faeture_region(const vector<Mat> &input, int region_size, int size1, int size2, int channels);
Mat get_scale_subwindow(const Mat &input, int cx, int cy, int width, int height, float *scaleFactors,
                        int *scale_model_sz, float currentScaleFactor, int nScales, int featureRatio, int num_hog_fea);
Mat new_gaussian_correlation(const vector<complex_mat> &xf, const vector<complex_mat> &yf, double sigma,
                             bool auto_correlation, int fea, int channel, int num_pca_fea);

model newtrainmodel(complex_mat &alphaf_num1, complex_mat &alphaf_num2, complex_mat &alphaf_den1,
                    complex_mat &alphaf_den2, float &d_num1, float &d_num2, float &d_den1, float &d_den2,
                    const Mat &k_hog, const Mat &k_cn, const complex_mat &yf, int frame, int start_frame,
                    float learning_rate_cn, float learning_rate_hog, const Mat &y, int imchannel, float lambda);
complex_mat resizeDFT2(const complex_mat &input, const int *sz);
complex_mat resizeDFT(const complex_mat &input, const int n);

float inference(std::string &input_dir, std::string &output_dir, std::string &item);