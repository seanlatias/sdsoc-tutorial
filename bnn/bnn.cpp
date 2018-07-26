#include "layer.h"
#include "conv2d.h"
#include "model.h"
#include "typedef.h"
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;

void bnn(bit input[MAX_FMAP], float output[10]){
  bit mem_conv1[MAX_FMAP];
  bit mem_conv2[MAX_FMAP];
  int mem_conv3[MAX_FMAP];

  /* First Conv Layer */
  pad(input, mem_conv1, 1, I_WIDTH1);
  conv_2d(mem_conv1, mem_conv3, 1, 32, 32, 0);
  batch_norm(mem_conv3, mem_conv1, miu1, sigma1, gamma1, beta1, 1, 32, I_WIDTH1);
  max_pool(mem_conv1, mem_conv2, 32, I_WIDTH1);

  /* Second Conv Layer */
  pad(mem_conv2, mem_conv1, 32, I_WIDTH2);
  conv_2d(mem_conv1, mem_conv3, 32, 64, 18, 1);
  batch_norm(mem_conv3, mem_conv1, miu2, sigma2, gamma2, beta2, 32, 64, I_WIDTH2);
  max_pool(mem_conv1, mem_conv2, 64, I_WIDTH2);

  reshape(mem_conv2, mem_conv1);

  float dense_input[3136];
  float dense_output[512];
  for (int i = 0; i < 3136; i++) dense_input[i] = mem_conv1[i].to_int();

  /* Dense Layers */
  dense(dense_input, dense_output, w_fc1, b_fc1, O_WIDTH*O_WIDTH*64, 512, true);
  dense(dense_output, output, w_fc2, b_fc2, 512, 10, false);
}
