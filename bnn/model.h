#ifndef MODEL_CONV
#define MODEL_CONV

#include"layer.h"

const bit w_conv1[MAX_W_CONV] = {
    #include"data/weight_0b"
}; //binary weight

const float miu1[MAX_F] = {
    #include"data/weight_3p"
};

const float sigma1[MAX_F] = {
    #include"data/weight_4p"
};

const float gamma1[MAX_F] = {
    #include"data/weight_1p"
};

const float beta1[MAX_F] = {
    #include"data/weight_2p"
};

const bit w_conv2[MAX_W_CONV] = {
    #include"data/weight_5b"
}; //binary weight

const float miu2[MAX_F] = {
    #include"data/weight_8p"
};

const float sigma2[MAX_F] = {
    #include"data/weight_9p"
};

const float gamma2[MAX_F] = {
    #include"data/weight_6p"
};

const float beta2[MAX_F] = {
    #include"data/weight_7p"
};

const float w_fc1[MAX_W_FC] = {
  #include"data/weight_10b"
};

const float b_fc1[FC2_UNITS] = {
  #include"data/weight_11p"
};

const float w_fc2[FC2_UNITS*OUT] = {
  #include"data/weight_12b"
};

const float b_fc2[OUT] = {
  #include"data/weight_13p"
};

#endif
