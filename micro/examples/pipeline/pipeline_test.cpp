/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <math.h>

#include "tensorflow/lite/core/c/common.h"
#include "pipeline_float_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
using PipelineOpResolver = tflite::MicroMutableOpResolver<1>;

TfLiteStatus RegisterOps(PipelineOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus ProfileMemoryAndLatency() {
  tflite::MicroProfiler profiler;
  PipelineOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 3000;
  uint8_t tensor_arena[kTensorArenaSize];
  constexpr int kNumResourceVariables = 24;

  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(g_pipeline_float_model_data), op_resolver, allocator,
      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
      &profiler);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);
  interpreter.input(0)->data.f[0] = 1.f;
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  MicroPrintf("");  // Print an empty new line
  profiler.LogTicksPerTagCsv();

  MicroPrintf("");  // Print an empty new line
  interpreter.GetMicroAllocator().PrintAllocations();
  return kTfLiteOk;
}

TfLiteStatus LoadFloatModelAndPerformInference() {
  const tflite::Model* model =
      ::tflite::GetModel(g_pipeline_float_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  PipelineOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 3000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  // Check if the predicted output is within a small range of the
  // expected output
  float epsilon = 0.05f;
  constexpr int kNumTestValues = 4;
  float golden_inputs[kNumTestValues] = {0.f, 1.f, 3.f, 5.f};

  for (int i = 0; i < kNumTestValues; ++i) {
    interpreter.input(0)->data.f[0] = golden_inputs[i];
    TF_LITE_ENSURE_STATUS(interpreter.Invoke());
    float y_pred = interpreter.output(0)->data.f[0];
    TFLITE_CHECK_LE(abs(sin(golden_inputs[i]) - y_pred), epsilon);
  }

  return kTfLiteOk;
}

int main(int argc, char* argv[]) {
  tflite::InitializeTarget();
  TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency());
  TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference());
  MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  return kTfLiteOk;
}
