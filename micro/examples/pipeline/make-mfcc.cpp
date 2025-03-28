/* ------------------------------------------------------------------
* Copyright (C) 2020 ewan xu<ewan_xu@outlook.com>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
* express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
* -------------------------------------------------------------------
*/

#include <librosa/librosa.h>

#include <iostream>
#include <mutex>
#include <fstream>  
#include <vector>
#include <iomanip>
#include <stdio.h>


#include <chrono>
#include <numeric>
#include <algorithm>

using namespace std;

/** 
  * This creates a mfcc matrix outputted as a string.  
  * Call it using the .wav audio source and 
  * you will get a matrix [[1,2], [3,4]] resembling a mfcc.
  *
  * @param x: input audio 
  * @param sr: input sample rate 
  * @return mfcc matrix 
*/

// Instead of taking in audio source, take in the pure audio file data

  std::vector<std::vector<float>> makeMfcc(std::vector<float> x, int sr, int num_mfcc, int num_mels){

  // Values opt for change incase of optimizing the performance
    int n_fft = 1024;
    int n_hop = 512;
    // Microphone used takes in audio from 20-20000 range 
    int fmin = 20;
    int fmax = 20000;
    string pad_mode = "reflect"; 
    // norm: Applying the last DCT transformation to make the mfcc 
    bool norm = true;
    // Amount of mfcc's, number of mels should stay the same  
    int n_mfcc = num_mfcc;
    int n_mels = num_mels;
    

    std::vector<std::vector<float>> mfcc_matrix = librosa::Feature::mfcc(x, sr, n_fft, n_hop, "hann", true, pad_mode, 2.f, n_mels, fmin, fmax, n_mfcc, norm, 2);


    // Transpose the mfcc matrix
    std::vector<std::vector<float>> mfcc_matrix_transposed(mfcc_matrix[0].size(), std::vector<float>(mfcc_matrix.size()));

    for (size_t i = 0; i < mfcc_matrix.size(); i++) {
        for (size_t j = 0; j < mfcc_matrix[i].size(); j++) {
            mfcc_matrix_transposed[j][i] = mfcc_matrix[i][j];
        }
    }


    return mfcc_matrix_transposed;
  }