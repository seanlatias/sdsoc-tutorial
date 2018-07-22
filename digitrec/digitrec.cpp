//==========================================================================
//digitrec.cpp
//==========================================================================
// @brief: A k-nearest-neighbor implementation for digit recognition

#include "digitrec.h"

// the following trick allows the use of macro in pragma
// ref. UG902 (v2014.1), page 55
#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)
#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 10
#endif

//----------------------------------------------------------
// Top function
//----------------------------------------------------------
// @param[in] : input - the testing instance
// @return : the recognized digit (0~9)

void digitrec( digit input, bit4& output ) 
{
  #include "training_data.h"

  // This array stores K minimum distances per training set
  bit6 knn_set[10][K_CONST];
  #pragma HLS array_partition variable=knn_set complete dim=1

  // Initialize the knn set
  for ( int i = 0; i < 10; ++i )
    for ( int k = 0; k < K_CONST; ++k )
      // Note that the max distance is 49
      knn_set[i][k] = 50; 

  for ( int i = 0; i < TRAINING_SIZE; ++i ) {
    #pragma HLS pipeline
    for ( int j = 0; j < 10; j++ ) {
      // UNROLL pragma with user-defined macro
      // DO_PRAGMA(HLS UNROLL factor=UNROLL_FACTOR) //UNROLL pragma with user-defined macro
      #pragma HLS array_partition variable=training_data block factor=10

      // Read a new instance from the training set
      digit training_instance = training_data[j * TRAINING_SIZE + i];
      // Update the KNN set
      update_knn( input, training_instance, knn_set[j] );
    }
  } 

  // Compute the final output
  knn_vote( knn_set, output ); 
}



//-----------------------------------------------------------------------
// update_knn function
//-----------------------------------------------------------------------
// Given the test instance and a (new) training instance, this
// function maintains/updates an array of K minimum
// distances per training set.

// @param[in] : test_inst - the testing instance
// @param[in] : train_inst - the training instance
// @param[in/out] : min_distances[K_CONST] - the array that stores the current
//                  K_CONST minimum distance values per training set

void update_knn( digit test_inst, digit train_inst, bit6 min_distances[K_CONST] )
{
  // Compute the difference using XOR
  digit diff = test_inst ^ train_inst;

  bit6 dist = 0;
  // Count the number of set bits
  for ( int i = 0; i < 49; ++i ) { 
  #pragma HLS UNROLL
    dist += diff[i];
  }

  bit6 max_dist = 0;
  int max_dist_id = K_CONST+1; 

  // Find the max distance
  for ( int k = 0; k < K_CONST; ++k ) {
    #pragma HLS UNROLL
    if ( min_distances[k] > max_dist ) {
      max_dist = min_distances[k];
      max_dist_id = k;
    }
  }
  
  // Replace the entry with the max distance
  if ( dist < max_dist )
    min_distances[max_dist_id] = dist;
}


//-----------------------------------------------------------------------
// knn_vote function
//-----------------------------------------------------------------------
// Given 10xK minimum distance values, this function 
// finds the actual K nearest neighbors and determines the
// final output based on the most common digit represented by 
// these nearest neighbors (i.e., a vote among KNNs). 
//
// @param[in] : knn_set - 10xK_CONST min distance values
// @return : the recognized digit
// 

void knn_vote( bit6 knn_set[10][K_CONST], bit4& min_index )
{
  min_index = 0;

  // This array keeps keeps of the occurences
  // of each digit in the knn_set
  
  int score[10]; 

  // Initialize score array  
  for ( int i = 0; i < 10; ++i )
      score[i] = 0; 

  // Find KNNs
  for ( int k = 0; k < K_CONST; ++k ) { 
    bit6 min_dist = 50;
    bit4 min_dist_id = 10;
    int  min_dist_record = K_CONST + 1;

    for ( int i = 0; i < 10; ++i ) {
        for (int k = 0; k < K_CONST; ++k ) {
        if ( knn_set[i][k] < min_dist ) {
          min_dist = knn_set[i][k];
          min_dist_id = i;
          min_dist_record = k;
        }
      }
    }
    
    score[min_dist_id]++;
    // Erase the minimum difference entry once it's recorded
    knn_set[min_dist_id][min_dist_record] = 50;
  }

  // Calculate the maximum score
  int max_score = 0; 
  for ( int i = 0; i < 10; ++i ) {
    if ( score[i] > max_score ) {
      max_score = score[i];
      min_index = i;
    }
  }
}

