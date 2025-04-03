#ifndef MAKE_MFCC_H
#define MAKE_MFCC_H

#include <string>
#include <vector>

// Function prototypes
std::vector<std::vector<float>> makeMfcc(std::vector<float> x, int sr, int num_mfcc, int num_mel);

void writeMfccToCsv(const std::vector<std::vector<float>>& mfcc_matrix, std::ofstream& outFile, std::string label);

std::vector<float> parseAudio(std::string audio_source);

#endif // MAKE-MFCC_H