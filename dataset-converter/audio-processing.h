#ifndef AUDIO_PROCESSING_H
#define AUDIO_PROCESSING_H

#include <vector>

/**
 * Pre-Emphasis.
 * 
 * This function applies pre-emphasis to the audio data.
 * 
 * @param input: Input audio data
 * @param alpha: Pre-emphasis coefficient (default: 0.97)
 * @return: Audio data after pre-emphasis
 */
std::vector<float> preEmphasis(const std::vector<float>& input, double alpha = 0.97);

/**
 * Compute RMS.
 * 
 * This function computes the Root Mean Square (RMS) of the audio data.
 * 
 * @param audio: Audio data
 * @return: RMS value of the audio data
 */
float computeRMS(const std::vector<float>& audio);

/**
 * RMS Normalize.
 * 
 * This function normalizes the audio data to a target RMS value.
 * 
 * @param audio: Audio data
 * @param targetRMS: Target RMS value (default: 0.1)
 * @return: Normalized audio data
 */
std::vector<float> rmsNormalize(const std::vector<float>& audio, float targetRMS = 0.1);

#endif // AUDIO_PROCESSING_H

