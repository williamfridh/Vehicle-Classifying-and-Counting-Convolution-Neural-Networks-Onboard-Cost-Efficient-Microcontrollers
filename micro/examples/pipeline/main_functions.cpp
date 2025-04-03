/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <vector>
#include <iostream>
#include "pico/stdlib.h"
#include "constants.h"
#include <fstream>  
#include <iomanip>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <malloc.h>
#include <librosa/librosa.h>
#include <cmath>


#include "pipeline_float_model_data.h"
#include "main_functions.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


// Globals, used for compatibility with Arduino-style sketches.
namespace {
  const uint8_t NUM_MFCC = 16;                          // 1 B
  const uint8_t NUM_MEL_BANDS = 32;                     // 1 B
  const uint16_t SAMPLE_RATE = 16000;                   // 2 B
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  uint16_t classifications[4] = {0};                    // 8 B

  constexpr uint16_t kTensorArenaSize = 37000;          // 2 B
  uint8_t tensor_arena[kTensorArenaSize];               // 37000 B
  
  std::vector<std::vector<float>> curMfcc;              // 512 B         
  std::vector<float> audioData;                        // 16000 B

  
                                                        // Total: 53.526 KB
}  // namespace
 
// Global variables, accessed by the main task.
bool setupError = false;

/**
 * Prints heap information.
 */
void printHeapInfo() {
  extern char __StackLimit, __bss_end__;
  uint32_t totalHeap = &__StackLimit - &__bss_end__;
  struct mallinfo m = mallinfo();
  uint32_t freeHeap = totalHeap - m.uordblks;

  printf("Heap used: %d bytes\n", (int)totalHeap);
  printf("Heap free: %d bytes\n", (int)freeHeap);
}

// The name of this function is important for Arduino compatibility.
void setup() {
  printHeapInfo();


  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_pipeline_float_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("Model provided is schema version %d not equal to supported version %d.\n", model->version(), TFLITE_SCHEMA_VERSION);
    setupError = true;
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)

  static tflite::MicroMutableOpResolver<7> resolver;

  TfLiteStatus resolve_status;
  resolve_status = resolver.AddConv2D();
  if (resolve_status != kTfLiteOk) {
      printf("Op resolver failed to add Conv2D\n");
      setupError = true;
      return;
  }

  resolve_status = resolver.AddFullyConnected();
  if (resolve_status != kTfLiteOk) {
      printf("Op resolver failed to add FullyConnected\n");
      setupError = true;
      return;
  }

  resolve_status = resolver.AddMaxPool2D();
  if (resolve_status != kTfLiteOk) {
      printf("Op resolver failed to add MaxPool2D\n");
      setupError = true;
      return;
  }

  resolve_status = resolver.AddSoftmax();
  if (resolve_status != kTfLiteOk) {
      printf("Op resolver failed to add Softmax\n");
      setupError = true;
      return;
  }

  resolve_status = resolver.AddReshape();
  if (resolve_status != kTfLiteOk) {
      printf("Op resolver failed to add Reshape\n");
      setupError = true;
      return;
  }

  resolve_status = resolver.AddMul();
  if (resolve_status != kTfLiteOk) {
      printf("Op resolver failed to add Mul\n");
      setupError = true;
      return;
  }

  resolve_status = resolver.AddAdd();
  if (resolve_status != kTfLiteOk) {
      printf("Op resolver failed to add Add\n");
      setupError = true;
      return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    printf("AllocateTensors() failed\n");
    setupError = true;
    return;
  }

  // Print how much memory is used by model.
  size_t bytes_used = interpreter->arena_used_bytes();
  size_t bytes_free = kTensorArenaSize - bytes_used;
  printf("Tensor arena size: %d bytes\n", kTensorArenaSize);
  printf("Bytes used: %d bytes\n", bytes_used);
  printf("Bytes free: %d bytes\n", bytes_free);

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Allocate memory for audio data
  audioData.resize(4000); // 4000 samples for 1 second of audio at 4000 Hz
  
  // Allocate memory for curMfcc
  curMfcc.resize(8); // 16 MFCCs
  for (int i = 0; i < 8; ++i) {
    curMfcc[i].resize(16); // 8 MFCC features
  }

  // Print out shape of input tensor
  printf("Input tensor shape: %d, %d, %d, %d\n", input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3]);

  // Print out shape of output tensor
  printf("Output tensor shape: %d, %d\n", output->dims->data[0], output->dims->data[1]);

}
 

int findClassificationIndex(){
  int output_size = output->dims->data[1]; // Assuming 1D output array
  int max_index = 0;
  float max_value = output->data.f[0];
  for (int i = 1; i < output_size; i++) {
    if (output->data.f[i] > max_value) {
      max_value = output->data.f[i];
      max_index = i;
    }
  }
  return max_index;
}

// Used to find the most classified vechicle 
int majorityVoting(){
  int tempVal = classifications[1];
  int correctClassification = 0;
  
  for(int i = 0; i < 4; i++){
    if(classifications[i] >= tempVal){
      tempVal = classifications[i];
      correctClassification = i;
    }
  }
  return correctClassification;
}

 /**
  * Pre-Emphasis.
  * 
  * This function applies pre-emphasis to the audio data.
  * 
  * @param input: Input audio data
  * @param audioBuffer: Buffer used for
  * @param alpha: Pre-emphasis coefficient (default: 0.97)
  * @return: Audio data after pre-emphasis
  */
 void preEmphasis(std::vector<float>& input, double alpha = 0.97) {
     if (input.empty()) return;  // Handle empty input case
 
     float tmp = input[0];
     for (size_t i = 1; i < input.size(); ++i) {
         input[i] = input[i] - alpha * tmp;
         tmp = input[i];
     }
 }
 
 /**
  * Compute RMS.
  * 
  * This function computes the Root Mean Square (RMS) of the audio data.
  * 
  * @param audio: Audio data
  * @return: RMS value of the audio data
  */
 float computeRMS(const std::vector<float>& audio) {
     float sum = 0.0;
     for (float sample : audio) {
         sum += sample * sample;
     }
     return sqrt(sum / audio.size());
 }
 
 /**
  * RMS Normalize.
  * 
  * This function normalizes the audio data to a target RMS value.
  * 
  * @param audio: Audio data
  * @param targetRMS: Target RMS value (default: 0.1)
  * @return: Nothing
  */
 void rmsNormalize(std::vector<float>& audio, float targetRMS = 0.1) {
 
     if (targetRMS < 0.1 || targetRMS > 0.3) {
         std::cerr << "Warning: Target RMS value should be between 0.1 and 0.3" << std::endl;
     }
 
     float currentRMS = computeRMS(audio);
     
     // Prevent division by zero
     if (currentRMS < 1e-8) {
         return; // Return original audio if the RMS is too small
     }
 
     float gain = targetRMS / currentRMS;
     
     for (size_t i = 0; i < audio.size(); i++) {
         audio[i] = audio[i] * gain;
     }
 }
 
 /**
  * Normalize to [-1, 1].
  * 
  * This function normalizes the audio data to the range [-1, 1].
  * 
  * @param audio: Audio data
  * @return: Nothing
  */
 void normalizeAudio(std::vector<float>& audio) {
     // Find max sample value
     float maxSample = 0.0;
     for (float sample : audio) {
         float tmp = std::abs(sample);
         if (maxSample < tmp) {
             maxSample = tmp;
         }
     }
     // Max samle too small, return without normalizing
     if (maxSample < 1e-8) {
         return;
     }
     // Normalize audio to [-1, 1]
     for (size_t i = 0; i < audio.size(); i++) {
         audio[i] = audio[i] / maxSample;
     }
 }
 
 /**
  * Aduio Processing.
  * 
  * This function makes all the calls to the seperate
  * audio processing functions.
  */
void audioProcessing(){
  normalizeAudio(audioData);
  rmsNormalize(audioData, 0.2);
  preEmphasis(audioData);
} 

// Collects to audio for the loop
void collectAudio(){
  int x_pointer = 0;
  while (x_pointer < 4000) {
    float value;
    if (scanf("%f", &value) == 1) {  // Read float from serial
      printf("You entered: %f\n", value);
      audioData[x_pointer] = value;
      x_pointer += 1;
    } else {
      //printf("Invalid input. Try again.\n");
      // Clear the buffer
      while (getchar() != '\n');
      return;
    }
  }

  //for(int i = 0; i < 4000; i++){
    // Collect audio data from the microphone
    // and overwrite whats in the audio buffer

    // Simulate audio data input
    //audioData[i] = (float)(rand() % (633 - 190 + 1) - 633);
  //}
}



/** 
  * This creates the mfcc matrix 

  * @param x: input audio 
  * @param sr: input sample rate 
  * @return mfcc matrix 
*/

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




void classifyAudio() {
  // Create MFCC
  printf("Creating MFCC\n");
  makeMfcc(curMfcc, audioData, SAMPLE_RATE, NUM_MFCC, NUM_MEL_BANDS);
  // Copy values directly into input tensor data with transposition
  printf("Populating input tensor...\n");
  for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 8; j++) {
          input->data.f[j * 16 + i] = curMfcc[j][i]; // Transpose by swapping indices
      }
  }
  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    printf("Invoke failed\n");
    return;
  }
  // Ensure output tensor is valid
  TfLiteTensor* output = interpreter->output(0);
  if (output == nullptr) {
    printf("Failed to get output tensor\n");
    return;
  }
  // Get classified index.
  int classificationIndex = findClassificationIndex();
  // Increment classification count
  classifications[classificationIndex]++;
  printf("Current classification: %d", classificationIndex);
  // If not background, reset background classification
  if (classificationIndex == 0) {
    classifications[0] = 0;
  }
}

// The name of this function is important for Arduino compatibility.
int iteraton = 0;
void loop() {
  // Print out heap information
  printHeapInfo();
  // Print out the current iteration
  printf("Iteration: %d\n\n", iteraton++);
  // Main function calls
  collectAudio();
  audioProcessing();
  classifyAudio();
  // If background is 5
  if (classifications[0] >= 5) {
    int maxClassification = majorityVoting();
    // Print out the classification result
    printf("Final classification: %d \n", maxClassification);
    // Reset classifications
    for (int i = 0; i < 4; i++) {
      classifications[i] = 0;
    }
  }
}

