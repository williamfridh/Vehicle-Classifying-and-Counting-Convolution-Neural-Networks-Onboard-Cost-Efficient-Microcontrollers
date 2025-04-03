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
#include "pipeline_float_model_data.h"
#include "main_functions.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <librosa/librosa.h>
#include "make-mfcc.h"
#include "audio-processing.h"






// Globals, used for compatibility with Arduino-style sketches.
namespace {
  const int NUM_MFCC = 16;
  const int NUM_MEL_BANDS = 32;
  const int SAMPLE_RATE = 16000;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  int classifications[4] = {0};

  constexpr int kTensorArenaSize = 37000;
  uint8_t tensor_arena[kTensorArenaSize];
  
  std::vector<std::vector<float>> curMfcc;
  std::vector<float> audioDataA;
  std::vector<float> audioDataB;
}  // namespace

// Global variables, accessed by the main task.
bool setupError = false;

// The name of this function is important for Arduino compatibility.
void setup() {
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

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Allocate memory for audio data
  audioDataA.resize(4000); // 4000 samples for 1 second of audio at 4000 Hz
  audioDataB.resize(4000); // 4000 samples for 1 second of audio at 4000 Hz
  
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

void audioProcessing(){
  normalizeAudio(audioDataA);
  rmsNormalize(audioDataA, 0.2);
  preEmphasis(audioDataA, audioDataB);
} 


void collectAudio(){
  for(int i = 0; i < 4000; i++){
    // Collect audio data from the microphone
    // and overwrite whats in the audio buffer

    // Simulate audio data input
    audioDataA[i] = (float)(rand() % (633 - 190 + 1) - 633);
  }

  audioProcessing();

}

void classifyAudio() {
  // Create MFCC
  makeMfcc(curMfcc, audioDataB, SAMPLE_RATE, NUM_MFCC, NUM_MEL_BANDS);
  // Copy values directly into input tensor data with transposition
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
  // If not background, reset background classification
  if (classificationIndex == 0) {
    classifications[0] = 0;
  }
}


// The name of this function is important for Arduino compatibility.
void loop() {
  collectAudio();
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

