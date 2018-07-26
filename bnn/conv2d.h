#ifndef CONV2D_H
#define CONV2D_H

#include "typedef.h"

#pragma SDS data access_pattern(input:SEQUENTIAL, output:SEQUENTIAL)
void conv_2d(bit input[MAX_FMAP], int output[MAX_FMAP], int M, int N, int I, int L);

#endif
