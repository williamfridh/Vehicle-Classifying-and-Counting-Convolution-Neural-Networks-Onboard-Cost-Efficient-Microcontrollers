/**
 * Audio Processing.
 * 
 * This file contains functions for audio processing.
 * These functions will be used for pre-processing audio data
 * for training machine learning models, but also on-board the microcontroller.
 */

#include <vector>
#include <cmath>
#include <iostream>

/**
 * Pre-Emphasis.
 * 
 * This function applies pre-emphasis to the audio data.
 * 
 * @param input: Input audio data
 * @param alpha: Pre-emphasis coefficient (default: 0.97)
 * @return: Audio data after pre-emphasis
 */
std::vector<float> preEmphasis(const std::vector<float>& input, double alpha = 0.97) {
    if (input.empty()) return {};  // Handle empty input case

    std::vector<float> output(input.size());
    output[0] = input[0];  // First sample remains unchanged

    for (size_t i = 1; i < input.size(); ++i) {
        output[i] = input[i] - alpha * input[i - 1];
    }

    return output;
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
 * @return: Normalized audio data
 */
std::vector<float> rmsNormalize(const std::vector<float>& audio, float targetRMS = 0.1) {

    if (targetRMS < 0.1 || targetRMS > 0.3) {
        std::cerr << "Warning: Target RMS value should be between 0.1 and 0.3" << std::endl;
    }

    float currentRMS = computeRMS(audio);
    
    // Prevent division by zero
    if (currentRMS < 1e-8) {
        return audio; // Return original audio if the RMS is too small
    }

    float gain = targetRMS / currentRMS;
    
    std::vector<float> normalizedAudio(audio.size());
    for (size_t i = 0; i < audio.size(); i++) {
        normalizedAudio[i] = audio[i] * gain;
    }
    
    return normalizedAudio;
}

