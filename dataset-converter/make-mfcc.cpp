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


#include "wavreader.h"
#include <librosa/librosa.h>

#include <iostream>
#include <vector>
#include <iomanip>
#include <stdio.h>


#include <chrono>
#include <numeric>
#include <algorithm>

using namespace std;



/** 
  * This takes in a matrix, parses it into a string
  *
  * @param mfcc matrix: The mfcc matrix 
  * @return mfcc matrix as a string 
*/

string mfccToString(std::vector<std::vector<float>> mfcc_matrix){
  std::string mfcc_string = "[";
  for (size_t i = 0; i < mfcc_matrix.size(); i++) {
      mfcc_string += "[";
      for (size_t j = 0; j < mfcc_matrix[i].size(); j++) {
          mfcc_string += std::to_string(mfcc_matrix[i][j]);
          if (j != mfcc_matrix[i].size() - 1) {
              mfcc_string += ", ";  // Add a comma between elements
          }
      }
      mfcc_string += "]";
      if (i != mfcc_matrix.size() - 1) {
          mfcc_string += ", ";  // Add a comma between rows
      }
  }
  mfcc_string += "]";

  return mfcc_string;
}



/** 
  * Parses the audio into samples which is used by mfcc
  *
  * @param audio_source:  The audio which is to be parsed 
  * @return The samples which are to be used by mfcc, and the sample_rate
*/

std::tuple<std::vector<float>, int> parseAudio(const char* audio_source){

  void* h_x = wav_read_open(audio_source);

  int format, channels, sr, bits_per_sample;
  unsigned int data_length;
  int res = wav_get_header(h_x, &format, &channels, &sr, &bits_per_sample, &data_length);
  if (!res)
  {
    cerr << "get ref header error: " << res << endl;
    //return -1;
  }

  int samples = data_length * 8 / bits_per_sample;
  std::vector<int16_t> tmp(samples);
  res = wav_read_data(h_x, reinterpret_cast<unsigned char*>(tmp.data()), data_length);
  if (res < 0)
  {
    cerr << "read wav file error: " << res << endl;
    //return -1;
  }
  std::vector<float> x(samples);
  std::transform(tmp.begin(), tmp.end(), x.begin(),
    [](int16_t a) {
    return static_cast<float>(a) / 32767.f;
  });

  return std::make_tuple(x, sr);
}



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

  std::vector<std::vector<float>> makeMfcc(std::vector<float> x, int sr){
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
    int n_mfcc = 25;
    int n_mels = 25;
    
    std::vector<std::vector<float>> mfcc_matrix = librosa::Feature::mfcc(x, sr, n_fft, n_hop, "hann", true, pad_mode, 2.f, n_mels, fmin, fmax, n_mfcc, norm, 2);
    return mfcc_matrix;
  }