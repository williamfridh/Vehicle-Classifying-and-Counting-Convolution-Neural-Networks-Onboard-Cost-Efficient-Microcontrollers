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

void makeMfcc(std::vector<std::vector<float>>& curMfcc, const std::vector<float>& x, int sr, int num_mfcc, int num_mel) {
  int n_fft = 1024;
  int n_hop = 512;
  int fmin = 20;
  int fmax = 20000;
  std::string pad_mode = "reflect";
  bool norm = true;
  int n_mfcc = num_mfcc;
  int n_mels = num_mel;

  // Directly save the result to curMfcc
  curMfcc = librosa::Feature::mfcc(x, sr, n_fft, n_hop, "hann", true, pad_mode, 2.f, n_mels, fmin, fmax, n_mfcc, norm, 2);
}
