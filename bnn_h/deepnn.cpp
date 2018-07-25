#include "layer.h"
#include "model_conv.h"
#include "typedef.h"
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;

void deepnn(bit x[I_WIDTH1 * I_WIDTH1], bit output[O_WIDTH*O_WIDTH * 64]){
  bit mem_conv1[MAX_FMAP] = {0};
  bit mem_conv2[MAX_FMAP] = {0};
  float mem_conv3[MAX_FMAP] = { 0 };

  pad(x, mem_conv1, 1, I_WIDTH1);

  conv_2d(mem_conv1, mem_conv3, w_conv1, 1, 32, 32, 1);
 
  batch_norm(mem_conv3, mem_conv1, miu1, sigma1, gamma1, beta1, 32, I_WIDTH1);

  max_pool(mem_conv1, mem_conv2, 32, I_WIDTH1);

  for (int i = 0; i < MAX_FMAP; i++) mem_conv1[i] = 0;

  pad(mem_conv2, mem_conv1, 32, I_WIDTH2);

  conv_2d(mem_conv1, mem_conv3, w_conv2, 32, 64, 18, 2);

  batch_norm(mem_conv3, mem_conv1, miu2, sigma2, gamma2, beta2, 64, I_WIDTH2);

  max_pool(mem_conv1, mem_conv2, 64, I_WIDTH2);
	
  for (int i = 0; i < O_WIDTH*O_WIDTH * 64; i++) output[i] = mem_conv2[i];
}
