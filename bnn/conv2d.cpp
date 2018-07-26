#include "layer.h"

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
void conv_2d(bit input[MAX_FMAP], int output[MAX_FMAP], int M, int N, int I, int L)
{
  int O = I - F + 1;
  int ifmap_size = I * I;
  int ofmap_size = O * O;

  static bit input_holder[MAX_FMAP];
  static int output_holder[MAX_FMAP];

  static const bit w_conv1[MAX_W_CONV] = {
    #include"data/weight_0b"
  };

  static const bit w_conv2[MAX_W_CONV] = {
    #include"data/weight_5b"
  };

  for (int i = 0; i < MAX_FMAP; i++) {
    input_holder[i] = input[i];
    output_holder[i] = 0;
  }

  for (int n = 0; n < N; n++){
    for (int m = 0; m < M; m++){
      for (int x = 0; x < O; x++){
        for (int y = 0; y < O; y++){
          int one_out = 0;
          int mac_num = 0;
          int o_index = x + y * O + n * ofmap_size;
          for (int c = 0; c < F; c++){
            for (int r = 0; r < F; r++){
              if (if_mac(x + c, y + r, I)) { //neglect padding pixels in mac
                int i_index = x + c + (y + r) * I + m * ifmap_size;
                int w_index = c + r * F + (n + m * N) * FILTER_SIZE;
                if (L == 0) one_out += input_holder[i_index] == w_conv1[w_index]; //XNOR
                else        one_out += input_holder[i_index] == w_conv2[w_index];
                mac_num++;
              }
            }
          }
          output_holder[o_index] += (one_out << 1) - mac_num;
        }
      }
    }
  }

  for (int i = 0; i < MAX_FMAP; i++) output[i] = output_holder[i];
}

