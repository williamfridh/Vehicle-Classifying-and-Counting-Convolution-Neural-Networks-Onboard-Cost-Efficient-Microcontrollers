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

#include "constants.h"
#include "pipeline_float_model_data.h"
#include "main_functions.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 300000;
  uint8_t tensor_arena[kTensorArenaSize];
  
  int x_pointer = 0; // Pointer to the current audio input data
  float x[640] = {0}; // Audio input data
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


  // Print out shape of input tensor
  printf("Input tensor shape: %d, %d, %d, %d\n", input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3]);

  // Print out shape of output tensor
  printf("Output tensor shape: %d, %d\n", output->dims->data[0], output->dims->data[1]);
}

// The name of this function is important for Arduino compatibility.
void loop() {

  //printf("Entering loop\n"); // Debugging statement

  // Read float from serial port
  float value;
  if (fread(&value, sizeof(float), 1, stdin) == 1) {  // Read float from serial
      //printf("You entered: %f\n", value);
      x[x_pointer] = value;
      x_pointer += 1;
  } else {
      //printf("Invalid input. Try again.\n");
      // Clear the buffer
      while (getchar() != '\n');
      return;
  }

  //printf("x_pointer: %d\n", x_pointer);

  if (x_pointer < 640) {
    return;
  }
  x_pointer = 0;

  // Reshape x_quantized to match the input tensor shape (1, 40, 16, 1)
  float x_quantized_reshaped[1][40][16][1];
  for (int i = 0; i < 40; i++) {
    for (int j = 0; j < 16; j++) {
      x_quantized_reshaped[0][i][j][0] = x[i * 16 + j];
    }
  }

  // Copy the audio input data to the input tensor
  memcpy(input->data.f, x_quantized_reshaped, sizeof(x_quantized_reshaped));

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

  // Output is an array, find the index of the largest value
  // Might be redudant if softmax is working
  int output_size = output->dims->data[1]; // Assuming 1D output array
  int max_index = 0;
  float max_value = output->data.f[0];
  for (int i = 1; i < output_size; i++) {
    if (output->data.f[i] > max_value) {
      max_value = output->data.f[i];
      max_index = i;
    }
  }
  //printf("Classification: %s\n", classes[max_index]);
  printf("%d\n", max_index);
}
