#ifndef MAKE_MFCC_H
#define MAKE_MFCC_H

#include <string>
#include <vector>

// Function prototypes
std::vector<std::vector<float>> makeMfcc(std::vector<float> x, int sr, int num_mfcc, int num_mels);

#endif // MAKE-MFCC_H