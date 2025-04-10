/**
 * Main functions for the audio classification pipeline.
 * 
 * This file contains the main functions for the audio classification pipeline.
 * It includes the setup and loop functions, as well as the audio processing functions.
 */

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

// Global variables to be used by other files.
bool setupError = false;

// File-specific global variables.
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  const uint8_t NUM_MFCC = 16;                          // 1 B
  const uint8_t NUM_MEL_BANDS = 32;                     // 1 B
  const uint16_t SAMPLE_RATE = 16000;                   // 2 B
  float softVotingPole[4] = {0};                        // 16 B
  uint8_t positive_streak = 0;                          // 1 B
  uint8_t negative_streak = 0;                          // 1 B

  constexpr uint32_t kTensorArenaSize = 37000;          // 2 B
  uint8_t tensor_arena[kTensorArenaSize];               // 37000 B
  
  std::vector<std::vector<float>> curMfcc;              // 512 B
  std::vector<float> audioData;                         // 16000 B

  const uint8_t NUM_CLASSES = 4;                        // 1 B
  const uint8_t negaticeClassIndex = 3;                 // 1 B

  std::vector<std::vector<float>> lastFiveSoftVotes;    // 80 B
  uint8_t lastFiveSoftVotesIndex = 0;                   // 1 B

                                                        // Total: 53.528 KB (UPDATE!)
}

/**
 * Prints heap information.
 * 
 * Prints the total heap size and the free heap size.
 */
void printHeapInfo() {
  extern char __StackLimit, __bss_end__;
  uint32_t totalHeap = &__StackLimit - &__bss_end__;
  struct mallinfo m = mallinfo();
  uint32_t freeHeap = totalHeap - m.uordblks;
  printf("s:Heap: %d / %d bytes\n", (int)totalHeap - (int)freeHeap, (int)totalHeap);
}

/**
 * Setup function.
 * 
 * This function initializes the TensorFlow Lite model and allocates memory for the tensors.
 * It also sets up the audio data and MFCC data structures.
 */
void setup() {
  // Print heap information at the initialization stage.
  printHeapInfo();

  // Initialize the TensorFlow Lite library.
  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_pipeline_float_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("e:Model provided is schema version %d not equal to supported version %d.\n", model->version(), TFLITE_SCHEMA_VERSION);
    setupError = true;
    return;
  }

  // Pull all custom operations needed (based on model) into the static resolver.
  static tflite::MicroMutableOpResolver<7> resolver;

  // Add operations to the resolver.
  if (resolver.AddConv2D() != kTfLiteOk || resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddMaxPool2D() != kTfLiteOk || resolver.AddSoftmax() != kTfLiteOk ||
      resolver.AddReshape() != kTfLiteOk || resolver.AddMul() != kTfLiteOk ||
      resolver.AddAdd() != kTfLiteOk) {
    printf("e:Op resolver failed to add required operations.\n");
    setupError = true;
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory for the model's tensors.
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    printf("e:AllocateTensors() failed\n");
    setupError = true;
    return;
  }

  // Print memory usage.
  size_t bytes_used = interpreter->arena_used_bytes();
  printf("s:Tensor arena size: %d/%d bytes\n", bytes_used, kTensorArenaSize);

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Allocate memory for audio data and MFCC matrix.
  audioData.resize(4000); // 4000 samples for 1 second of audio at 4000 Hz
  curMfcc.resize(8, std::vector<float>(16)); // 8x16 MFCC matrix
  lastFiveSoftVotes.resize(5, std::vector<float>(NUM_CLASSES)); // 5x4 soft voting matrix
  for (int i = 0; i < 5; i++) {
    lastFiveSoftVotes[i].resize(NUM_CLASSES, 0); // Initialize soft voting matrix
  }

  // Print input and output tensor shapes.
  printf("s:Input tensor shape: %d, %d, %d, %d\n", input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3]);
  printf("s:Output tensor shape: %d, %d\n", output->dims->data[0], output->dims->data[1]);
}

/**
 * Pre-Emphasis.
 * 
 * This function applies pre-emphasis to the audio data.
 * 
 * @param input: Input audio data
 * @param alpha: Pre-emphasis coefficient (default: 0.97)
 * @return: Audio data after pre-emphasis
 */
void preEmphasis(std::vector<float>& input, double alpha = 0.97) {
  //if (input.empty()) return;  // Handle empty input case

  float prev = input[0];  // Store the first sample
  for (size_t i = 1; i < input.size(); ++i) {
    float current = input[i];
    input[i] = input[i] - alpha * prev;  // Apply the pre-emphasis formula
    prev = current;  // Update the previous sample
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
  //if (targetRMS < 0.1 || targetRMS > 0.3) {
  //  std::cerr << "w: Target RMS value should be between 0.1 and 0.3" << std::endl;
  //}

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
  // Max sample too small, return without normalizing
  if (maxSample < 1e-8) {
    return;
  }
  // Normalize audio to [-1, 1]
  for (size_t i = 0; i < audio.size(); i++) {
    audio[i] = audio[i] / maxSample;
  }
}

/**
 * Collect Audio.
 */
void collectAudio() {
  int x_pointer = 0;
  while (x_pointer < 4000) {
    float value;
    if (fread(&value, sizeof(float), 1, stdin) == 1) {  // Read float from binary input
      audioData[x_pointer] = value;
      x_pointer += 1;
    } else {
      printf("e:Invalid input or end of stream. Exiting collection.\n");
      return;
    }
  }
}

/**
 * Collect Audio.
 */
void collectAudioFramesUSB() {
  int x_pointer = 0;
  while (x_pointer < 128) {
    float value;
    if (fread(&value, sizeof(float), 1, stdin) == 1) {  // Read float from binary input
      audioData[x_pointer] = value;
      x_pointer += 1;
    } else {
      printf("e:Invalid input or end of stream. Exiting collection.\n");
      return;
    }
  }
}

/**
 * Generate random audio data.
 * 
 * Populates the audioData vector with random values between -1 and 1 (float32).
 */
void generateRandomAudioData() {
  for (int i = 0; i < audioData.size(); ++i) {
    audioData[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Random float between -1 and 1
  }
}

/**
 * Create MFCC matrix.
 * 
 * @param curMfcc: Output MFCC matrix
 * @param x: Input audio
 * @param sr: Input sample rate
 * @param num_mfcc: Number of MFCC features
 * @param num_mel: Number of Mel bands
 */
void makeMfcc(std::vector<std::vector<float>>& curMfcc, const std::vector<float>& x, int sr, int num_mfcc, int num_mel) {
  int n_fft = 1024;
  int n_hop = 512;
  int fmin = 20;
  int fmax = 8000;
  std::string pad_mode = "reflect";
  bool norm = true;
  int n_mfcc = num_mfcc;
  int n_mels = num_mel;

  curMfcc = librosa::Feature::mfcc(x, sr, n_fft, n_hop, "hann", true, pad_mode, 2.f, n_mels, fmin, fmax, n_mfcc, norm, 2);
}

/**
 * Check if class is positive.
 * 
 * This function checks if the class index is positive.
 * 
 * @param classIndex: Class index to check
 * @return: True if class index is positive, false otherwise
 */
bool classIsPositive(int classIndex) {
  return classIndex != negaticeClassIndex;
}

/**
 * Classify Audio.
 * 
 * This function classifies the audio data using the TensorFlow Lite model.
 * 
 * @return: Classification index
 */
int classifyAudio() {
  // Create MFCC.
  makeMfcc(curMfcc, audioData, SAMPLE_RATE, NUM_MFCC, NUM_MEL_BANDS);

  // Transpose and populate input tensor.
  float x_quantized_reshaped[1][16][8][1];
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 8; j++) {
      x_quantized_reshaped[0][i][j][0] = curMfcc[j][i];
    }
  }
  memcpy(input->data.f, x_quantized_reshaped, sizeof(x_quantized_reshaped));

  // Run inference.
  if (interpreter->Invoke() != kTfLiteOk) {
    printf("e:Invoke failed\n");
    return -1;
  }

  // Increment soft voting pole and store the output as one of the last five votes.
  for (int i = 0; i < NUM_CLASSES; i++) {
    softVotingPole[i] += output->data.f[i];
    lastFiveSoftVotes[lastFiveSoftVotesIndex][i] = output->data.f[i];
  }
  lastFiveSoftVotesIndex = (lastFiveSoftVotesIndex + 1) % 5;

  // Get classification index.
  int lastVoteWinner = 0;
  float max_value = output->data.f[0];
  for (int i = 0; i < NUM_CLASSES; i++) {
    if (output->data.f[i] > max_value) {
      max_value = output->data.f[i];
      lastVoteWinner = i;
    }
  }

  // Print current soft voting pole and last vote winner.
  printf("c: [%f,%f,%f,%f] voted for: %d\n", softVotingPole[0], softVotingPole[1], softVotingPole[2], softVotingPole[3], lastVoteWinner);

  return lastVoteWinner;
}

/**
 * Finalize Classification.
 * 
 * This function finalizes the classification based on the majority voting result.
 * 
 * @param majorityVoting: Majority voting result
 */
void finalizeClassification(int majorityVoting) {
  printf("v:%d\n", majorityVoting);
  positive_streak = 0;
  negative_streak = 0;
  for (int i = 0; i < NUM_CLASSES; i++) {
    softVotingPole[i] = 0;
  }
  // Sum last five soft votes and set soft voting pole to the sum.
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < NUM_CLASSES; j++) {
      softVotingPole[j] += lastFiveSoftVotes[i][j];
    }
  }
}

// The name of this function is important for Arduino compatibility.
int iteration = 0;
void loop() {
  collectAudio();
  //collectAudioFramesUSB();
  normalizeAudio(audioData);
  rmsNormalize(audioData, 0.2);
  preEmphasis(audioData);

  int classificationIndex = classifyAudio();
  if (classificationIndex == -1) {
    printf("e:Classification failed\n");
    return;
  }


  // Majority voting
  float maxValue = softVotingPole[0];
  int majorityVote = 0;
  for (int i = 0; i < NUM_CLASSES; i++) {
    //if (i == negaticeClassIndex) {
    //  continue; // Skip negative class index
    //}
    if (softVotingPole[i] > maxValue) {
      maxValue = softVotingPole[i];
      majorityVote = i;
    }
  }

  if (classIsPositive(classificationIndex)) {
    positive_streak++;
    if (positive_streak >= 2) {
      negative_streak = 0;
    }
    if (positive_streak >= 5) {
      if (!classIsPositive(majorityVote)) {
        finalizeClassification(majorityVote);
      }
    }
  } else {
    negative_streak++;
    if (negative_streak >= 2) {
      positive_streak = 0;
    }
    if (negative_streak >= 5) {
      if (classIsPositive(majorityVote)) {
        finalizeClassification(majorityVote);
      }
    }
  }
}
