#ifndef LAYER_H
#define LAYER_H

#include"typedef.h"

const int I_WIDTH1 = 16; //conv1 input width
const int I_WIDTH2 = 8; //conv2 input width
const int O_WIDTH = 4; //conv output width
const int F = 3; //filter width
const int FILTER_SIZE = F*F;
const int PADDING = F - 1;
const int MAX_FMAP = 819200; //10*10*64*128
const int MAX_F = 128; //{64, 128} num of conv2 output fmaps
const int MAX_W_CONV = 73728;//3*3*64*128 num of conv2 weights
const int FC1_UNITS = O_WIDTH*O_WIDTH * 128; //num of fc1 input units
const int FC2_UNITS = 512;
const int MAX_W_FC = FC1_UNITS*FC2_UNITS;
const int OUT = 10;

void pad(bit input[MAX_FMAP], bit output[MAX_FMAP], int M, int I);

void max_pool(bit input[MAX_FMAP], bit output[MAX_FMAP], int M, int I);

void batch_norm(int input[MAX_FMAP], bit output[MAX_FMAP], const float miu[MAX_F], const float sigma[MAX_F], const float gamma[MAX_F], const float beta[MAX_F], int M0, int M, int I);

void reshape(bit* input, bit* output);

void dense(float* input, float*output, const float*weight, const float* bias, int M, int N, bool use_relu);

#endif
