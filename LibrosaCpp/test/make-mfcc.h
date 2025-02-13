#ifndef MAKE-MFCC_H
#define MAKE-MFCC_H

#include <string>
#include <vector>

// Function prototypes
std::string makeMfcc(std::string audio_source);

std::string mfccToString(std::vector<std::vector<float>> mfcc_matrix);

std::vector<float> parseAudio(std::string audio_source);

#endif // MAKE-MFCC_H