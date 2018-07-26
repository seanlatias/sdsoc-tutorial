#include <iostream>
#include <fstream>
#include "bnn.h"

const int TEST_SIZE = 10;

void read_test_images(int8_t test_images[TEST_SIZE][784]) {
  std::ifstream infile("data/test_b.dat");
  if (infile.is_open()) {
    for (int index = 0; index < TEST_SIZE; index++) {
      for (int pixel = 0; pixel < 784; pixel++) {
        int i;
        infile >> i;
        test_images[index][pixel] = i;
      }
    }
    infile.close();
  }
}

void read_test_labels(int test_labels[TEST_SIZE]) {
  std::ifstream infile("data/label.dat");
  if (infile.is_open()) {
    for (int index = 0; index < TEST_SIZE; index++) {
      infile >> test_labels[index];
    }
    infile.close();
  }
}


int main(){

  int8_t test_images[TEST_SIZE][784];
  int test_labels[TEST_SIZE];
  read_test_images(test_images);
  read_test_labels(test_labels);

  float correct = 0.0;
  
  for (int test = 0; test < TEST_SIZE; test++) {

    bit input_image[I_WIDTH1*I_WIDTH1];
    float output[10];

    for (int i = 0; i < 784; i++)
      input_image[i] = test_images[test][i];

    bnn(input_image, output);

    int max_id = 0;
    for(int i = 1; i < 10; i++)
      if(output[i] > output[max_id])
        max_id = i;
    if (max_id == test_labels[test]) correct += 1.0;
    //cout << test << ": " << max_id << " " << test_labels[test] << endl;
  }
  std::cout << correct/TEST_SIZE << std::endl;
  
	return 0;
}
