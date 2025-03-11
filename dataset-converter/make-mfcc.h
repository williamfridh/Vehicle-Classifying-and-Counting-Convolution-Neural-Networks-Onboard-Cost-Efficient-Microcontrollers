#ifndef MAKE_MFCC_H
#define MAKE_MFCC_H

#include <string>
#include <vector>

// Function prototypes
std::vector<std::vector<float>> makeMfcc(std::vector<float> x, int sr);

std::string mfccToString(std::vector<std::vector<float>> mfcc_matrix);

std::vector<float> parseAudio(std::string audio_source);

#endif // MAKE-MFCC_H