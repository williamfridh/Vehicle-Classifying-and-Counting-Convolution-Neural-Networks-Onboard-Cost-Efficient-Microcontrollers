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
  int softVotingPool[4] = {0};                          // 16 B

  constexpr uint32_t kTensorArenaSize = 37000;          // 2 B
  uint8_t tensor_arena[kTensorArenaSize];               // 37000 B
  
  std::vector<std::vector<float>> curMfcc;              // 512 B
  std::vector<float> audioData;                         // 16000 B

  const uint8_t NUM_CLASSES = 4;                        // 1 B
  const uint8_t negaticeClassIndex = 0;                 // 1 B

  std::vector<std::vector<int>> lastXSoftVotes;         // ? B
  int8_t lastXSoftVotesIndex = 0;                       // 1 B
  int8_t lastXVoteAmount = 4;                           // 1 B
  int8_t minVotesBeforeClassify = 5;                     // 1 B

  uint8_t lastVotePostitive = 2;                        // 1 B

  const float mfccMean = -5.2690635;
  const float mfccStd = 15.966296;

  int64_t total_elapsed_us1 = 0;
  int64_t total_elapsed_us2 = 0;
  int64_t total_elapsed_us3 = 0;
  int64_t total_elapsed_us4 = 0;
  int64_t total_elapsed_us5 = 0;
  int64_t total_elapsed_us6 = 0;
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
  lastXSoftVotes.resize(lastXVoteAmount, std::vector<int>(NUM_CLASSES)); // X*4 soft voting matrix
  for (int i = 0; i < lastXVoteAmount; i++) {
    lastXSoftVotes[i].resize(NUM_CLASSES, 0); // Initialize soft voting matrix
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
  size_t bytesRead = fread(audioData.data(), sizeof(float), audioData.size(), stdin); // Read array of floats directly into audioData
  if (bytesRead != audioData.size()) {
    printf("e:Invalid input or end of stream. Exiting collection.\n");
    return;
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


int8_t normalize_and_quantize(float x) {
  // 1. Normalize: divide by 3
  float normalized = x / 3.0f;
  
  // 2. Scale to 127
  float scaled = normalized * 127.0f;

  // 3. Round to nearest integer (ties to even, like numpy)
  float rounded = roundf(scaled);

  // 4. Clip to int8_t range [-128, 127]
  if (rounded > 127.0f) {
      rounded = 127.0f;
  } else if (rounded < -128.0f) {
      rounded = -128.0f;
  }

  // 5. Cast to int8_t
  return (int8_t)rounded;
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
  int8_t x_quantized_reshaped[1][16][8][1];
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 8; j++) {
      float mfcc = (curMfcc[j][i] - mfccMean) / mfccStd;
      if (mfcc < -3.0f) {
        mfcc = -3.0f;
      } else if (mfcc > 3.0f) {
        mfcc = 3.0f;
      }
      int8_t quantized = normalize_and_quantize(mfcc);
      x_quantized_reshaped[0][i][j][0] = quantized;
    }
  }
  
  memcpy(input->data.int8, x_quantized_reshaped, sizeof(x_quantized_reshaped));
  // Measure the time it takes to run inference.
  //auto start = std::chrono::high_resolution_clock::now();

  absolute_time_t start6 = get_absolute_time();

  // Run inference.
  if (interpreter->Invoke() != kTfLiteOk) {
    printf("e:Invoke failed\n");
    return -1;
  }
  
  absolute_time_t end6 = get_absolute_time();
  int64_t elapsed_us6 = absolute_time_diff_us(start6, end6);
  total_elapsed_us6 += elapsed_us6;

  // Increment soft voting pool and store the output as one of the last five votes.
  for (int i = 0; i < NUM_CLASSES; i++) {
    softVotingPool[i] += output->data.int8[i];
    lastXSoftVotes[lastXSoftVotesIndex][i] = output->data.int8[i];
  }
  lastXSoftVotesIndex = (lastXSoftVotesIndex + 1) % lastXVoteAmount;

  // Get classification index.
  int lastVoteWinner = 0;
  int8_t max_val = output->data.int8[0];  // Use int8_t
  for (int i = 0; i < NUM_CLASSES; i++) {
    if (output->data.int8[i] > max_val) {
      max_val = output->data.int8[i];
      lastVoteWinner = i;
    }
  }

  printf("c: [%d,%d,%d,%d] voted for: %d max value: %d \n", softVotingPool[0], softVotingPool[1], softVotingPool[2], softVotingPool[3], lastVoteWinner, max_val);

  return lastVoteWinner;
}

/**
 * Finalize Classification.
 * 
 * This function finalizes the classification based on the plurality voting result.
 * 
 * @param pluralityVoting: Plurality voting result
 */
void finalizeClassification(int pluralityVoteMinusFiveLast, int pluralityVoteLastFive) {

  if (lastVotePostitive == 2) {
    lastVotePostitive = classIsPositive(pluralityVoteMinusFiveLast) ? 1 : 0;
  } else {
    lastVotePostitive = lastVotePostitive == 1 ? 0 : 1;
  }

  printf("v:%d\n", pluralityVoteMinusFiveLast);
  for (int i = 0; i < NUM_CLASSES; i++) {
    softVotingPool[i] = 0;
  }
  // Sum last five soft votes and set soft voting pole to the sum.
  for (int i = 0; i < lastXVoteAmount; i++) {
    for (int j = 0; j < NUM_CLASSES; j++) {
      softVotingPool[j] += lastXSoftVotes[i][j];
    }
  }
}



// The name of this function is important for Arduino compatibility.
int totalLoops = 0;
void loop() {
  // Start timestamp
  absolute_time_t start1 = get_absolute_time();
  
  collectAudio();

  absolute_time_t end1 = get_absolute_time();
  int64_t elapsed_us1 = absolute_time_diff_us(start1, end1);
  total_elapsed_us1 += elapsed_us1;
  //collectAudioFramesUSB();

  absolute_time_t start2 = get_absolute_time();
  
  normalizeAudio(audioData);

  absolute_time_t end2 = get_absolute_time();
  int64_t elapsed_us2 = absolute_time_diff_us(start2, end2);
  total_elapsed_us2 += elapsed_us2;

  absolute_time_t start3 = get_absolute_time();

  rmsNormalize(audioData, 0.2);

  absolute_time_t end3 = get_absolute_time();
  int64_t elapsed_us3 = absolute_time_diff_us(start3, end3);
  total_elapsed_us3 += elapsed_us3;

  absolute_time_t start4 = get_absolute_time();

  preEmphasis(audioData);

  absolute_time_t end4 = get_absolute_time();
  int64_t elapsed_us4 = absolute_time_diff_us(start4, end4);
  total_elapsed_us4 += elapsed_us4;

  int classificationIndex = classifyAudio();
  if (classificationIndex == -1) {
    printf("e:Classification failed\n");
    return;
  }

  absolute_time_t start5 = get_absolute_time();

  // Combine five last soft votes and set soft voting pole to the sum.
  int lastXSoftVotesCombined[NUM_CLASSES] = {0};
  for (int i = 0; i < lastXVoteAmount; i++) {
    for (int j = 0; j < NUM_CLASSES; j++) {
      lastXSoftVotesCombined[j] += lastXSoftVotes[i][j];
    }
  }

  // Get plurality vote from the last five soft votes.
  int pluralityVoteLastFive = 0;
  int max_val = lastXSoftVotesCombined[0];  // Use int8_t
  for (int i = 0; i < NUM_CLASSES; i++) {
    if (lastXSoftVotesCombined[i] > max_val) {
      max_val = lastXSoftVotesCombined[i];
      pluralityVoteLastFive = i;
    }
  }

  // Copy softVoteing pool to last five soft votes.
  int softVotePoolMinusFiveLast[NUM_CLASSES] = {0};
  for (int i = 0; i < NUM_CLASSES; i++) {
    softVotePoolMinusFiveLast[i] = softVotingPool[i] - lastXSoftVotesCombined[i];
  }

  // Get plurality vote from the soft voting pool minus last five votes.
  int pluralityVoteMinusFiveLast = 0;
  max_val = softVotePoolMinusFiveLast[0];  // Use int8_t
  for (int i = 0; i < NUM_CLASSES; i++) {
    if (softVotePoolMinusFiveLast[i] > max_val) {
      max_val = softVotePoolMinusFiveLast[i];
      pluralityVoteMinusFiveLast = i;
    }
  }

  // Here we determine if we have a new classification.
  // If the classification of the last votes (minus last five votes) is different
  // from the last five votes, we have a new classification. But this should only be
  // sent to finalizeClassification if the last five votes as the last votes minus
  // the last five votes. Meaning, it wont classify unless it detects a switch in audio.
  if (
    ( classIsPositive(pluralityVoteMinusFiveLast) != lastVotePostitive ||
    lastVotePostitive == 2 ) &&
    classIsPositive(pluralityVoteMinusFiveLast) != classIsPositive(pluralityVoteLastFive) &&
    minVotesBeforeClassify == 0
  ) {
    finalizeClassification(pluralityVoteMinusFiveLast, pluralityVoteLastFive);
  } else {
    if (minVotesBeforeClassify > 0) {
      minVotesBeforeClassify--;
    }
  }

  absolute_time_t end5 = get_absolute_time();
  int64_t elapsed_us5 = absolute_time_diff_us(start5, end5);
  total_elapsed_us5 += elapsed_us5;

  if (totalLoops % 1000 == 0 && totalLoops > 0)
  {
    double avg_elapsed_us1 = total_elapsed_us1 / (double)totalLoops;
    double avg_elapsed_us2 = total_elapsed_us2 / (double)totalLoops;
    double avg_elapsed_us3 = total_elapsed_us3 / (double)totalLoops;
    double avg_elapsed_us4 = total_elapsed_us4 / (double)totalLoops;
    double avg_elapsed_us5 = total_elapsed_us5 / (double)totalLoops;
    double avg_elapsed_us6 = total_elapsed_us6 / (double)totalLoops;

    printf("c:===== AVERAGE ELAPSED TIMES OVER %d RUNS =====\n", totalLoops);
    printf("c:Average Collecting Audio:       %.2f us\n", avg_elapsed_us1);
    printf("c:Average normalize Audio:       %.2f us\n", avg_elapsed_us2);
    printf("c:Average RMS normalize Audio:       %.2f us\n", avg_elapsed_us3);
    printf("c:Average preemphesis Audio:    %.2f us\n", avg_elapsed_us4);
    printf("c:Average voting Audio:      %.2f us\n", avg_elapsed_us5);
    printf("c:Average Classifying Audio(invoke):      %.2f us\n", avg_elapsed_us6);
    printf("fin:");
  }
  totalLoops++;
}
