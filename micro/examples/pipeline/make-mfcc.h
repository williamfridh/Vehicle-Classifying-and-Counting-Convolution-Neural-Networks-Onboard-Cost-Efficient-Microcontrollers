#ifndef MAKE_MFCC_H
#define MAKE_MFCC_H

#include <string>
#include <vector>

// Function prototypes
std::vector<std::vector<float>> makeMfcc(std::vector<float> x, int sr, int num_mfcc);

#endif // MAKE-MFCC_H