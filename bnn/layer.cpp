#include "layer.h"
#include "typedef.h"
#include <cmath>
#include <iostream>

using namespace std;

void pad(bit input[MAX_FMAP], bit output[MAX_FMAP], int M, int I) {
  int ifmap_size = I * I;
  int ofmap_size = (I+PADDING) * (I+PADDING);

  for (int i = 0; i < MAX_FMAP; i++) output[i] = 0;

  for (int m = 0; m < M; m++) {
    for (int x = 0; x < I; x++) {
      for (int y = 0; y < I; y++) {
        int i_index = x + y*I + m*ifmap_size;
        int o_index = (x + PADDING/2) + (y + PADDING/2)*(I + PADDING) + m*ofmap_size;
        output[o_index] = input[i_index];
      }
    }
  }
}

void max_pool(bit input[MAX_FMAP], bit output[MAX_FMAP], int M, int I){
  int O = I / 2;
  int ifmap_size = I * I;
  int ofmap_size = O * O;

  for (int i = 0; i < MAX_FMAP; i++) output[i] = 0;

  for (int m = 0; m < M; m++){
    for (int x = 0; x < O; x++){
      for (int y = 0; y < O; y++){
        int o_index = x + y * O + m * ofmap_size;
        bit max = 0;
        for (int c = 0; c < 2; c++){
          for (int r = 0; r < 2; r++){
            int i_index = 2 * x + c + (2 * y + r) * I + m * ifmap_size;
            if (input[i_index]) max = 1; //this is because bit 1 is represented as 0xff memory
          }
        }
        output[o_index] = max;
      }
    }
  }
}

void batch_norm(int input[MAX_FMAP], bit output[MAX_FMAP], const float miu[MAX_F], const float sigma[MAX_F], const float gamma[MAX_F], const float beta[MAX_F], int M0, int M, int I){
  int ifmap_size = I * I;

  float var_w = 2. / (F*F * M0);
  float con = sqrt(var_w);

  for (int m = 0; m < M; m++){
    float s = sqrt(sigma[m] + 0.00001);
    float k = con * gamma[m] / s;
		float h = -miu[m] * gamma[m] / s + beta[m];
    for (int x = 0; x < I; x++){
      for (int y = 0; y < I; y++){
        int index = x + y * I + m * ifmap_size;
        output[index] = (((float)input[index] * k + h > 0) ? 1 : 0); //quantize
      }
    }
  }
}

void reshape(bit* input, bit* output) {
  for (int c = 0; c < 32; c++) {
    for (int y = 0; y < 4; y++) {
      for (int x = 0; x < 4; x++) {
        int o_index = c + (x + y * 4 ) * 32;
        int i_index = x + y * 4 + c * 16;
        output[o_index] = input[i_index];
      }
    }
  }
}

void dense(float* input, float* output, const float* weight, const float* bias, int M, int N, bool use_relu){
  float var_w = 2. / M;
  float c = sqrt(var_w);

  for (int n = 0; n < N; n++){
		float one_out = 0;
    for (int m = 0; m < M; m++) {
      int w_index = m * N + n;
			one_out += input[m] == weight[w_index]; //XNOR
    }
		output[n] = (2 * one_out - M)*c;
    float biased = output[n] + bias[n];
    if (use_relu) output[n] = (biased > 0) ? 1 : 0;
    else output[n] = biased;
  }

}


