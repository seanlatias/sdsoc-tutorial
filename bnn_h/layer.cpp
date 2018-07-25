#include "layer.h"
#include "typedef.h"
#include <cmath>
#include <iostream>

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

inline bool if_mac(int x, int y, int I)
{
	if (x < PADDING / 2 || x >= (I - PADDING / 2) || y < PADDING / 2 || y >= (I - PADDING / 2))
		return false;
	return true;
}

// @param[in] : input - input fmaps
//              weight - filters
//              M - number of input fmaps
//              N - number of output fmaps
//              I - width of input fmaps
// @param[out] : output - output fmaps
void conv_2d(bit input[MAX_FMAP], float output[MAX_FMAP], const bit weight[MAX_W_CONV], int M, int N, int I, int layer)
{
	int O = I - F + 1;
	int ifmap_size = I * I;
	int ofmap_size = O * O;

	float var_w = 2. / (F*F * M);
	float con = sqrt(var_w);

	static bit input_tmp[MAX_FMAP];
	static float output_tmp[MAX_FMAP];
	static bit weight_tmp[MAX_W_CONV];

	const bit w_conv1[MAX_W_CONV] = {
		#include"data/weight_0b"
	}; //binary weight

	const bit w_conv2[MAX_W_CONV] = {
		#include"data/weight_5b"
	}; //binary weight

	for (int i = 0; i < MAX_FMAP; i++) {
		input_tmp[i] = input[i];
		output_tmp[i] = 0;
	}

	for (int i = 0; i < MAX_W_CONV; i++) {
		weight_tmp[i] = weight[i];
	}

	for (int n = 0; n < N; n++){
		for (int m = 0; m < M; m++){
			for (int x = 0; x < O; x++){
				for (int y = 0; y < O; y++){
					float one_out = 0;
					int mac_num = 0;
					int o_index = x + y * O + n * ofmap_size;
					for (int c = 0; c < F; c++){
						for (int r = 0; r < F; r++){
							if (if_mac(x + c, y + r, I)) { //neglect padding pixels in mac
								int i_index = x + c + (y + r) * I + m * ifmap_size;
								int w_index = c + r * F + (n + m * N) * FILTER_SIZE;
								if (layer == 1)
									one_out += ~(input_tmp[i_index] ^ w_conv1[w_index]); //XNOR
								else
									one_out += ~(input_tmp[i_index] ^ w_conv2[w_index]);
								mac_num++;
							}
						}
					}
					output_tmp[o_index] += (2 * one_out - mac_num)*con;
				}
			}
		}
	}

	for (int i = 0; i < MAX_FMAP; i++)
		output[i] = output_tmp[i];
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
						if (input[i_index] == -1) max = input[i_index]; //this is because bit 1 is represented as 0xff memory
					}
				}
				output[o_index] = max;
			}
		}
	}
}

void batch_norm(float input[MAX_FMAP], bit output[MAX_FMAP], const float miu[MAX_F], const float sigma[MAX_F], const float gamma[MAX_F], const float beta[MAX_F], int M, int I){
	int ifmap_size = I * I;

	for (int m = 0; m < M; m++){
		float s = sqrt(sigma[m] + 0.00001);
		float k = gamma[m] / s;
		float h = -miu[m] * gamma[m] / s + beta[m];
		for (int x = 0; x < I; x++){
			for (int y = 0; y < I; y++){
				int index = x + y * I + m * ifmap_size;
				output[index] = ((input[index] * k + h > 0) ? 1 : 0); //quantize
			}
		}
	}
}

void reshape(float* input, float* output) {
	for (int c = 0; c < 64; c++) {
		for (int y = 0; y < 7; y++) {
			for (int x = 0; x < 7; x++) {
				int o_index = c + (x + y * 7 ) * 64;
				int i_index = x + y * 7 + c * 49;
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
      //output[n] += input[m] * weight[w_index] * c;
			one_out += (input[m] == weight[w_index]) ? 1 : 0; //XNOR
    }
		output[n] = (2 * one_out - M)*c;
    float biased = output[n] + bias[n];
    if (use_relu) output[n] = (biased > 0) ? 1 : 0;
    else output[n] = biased;
  }

}
