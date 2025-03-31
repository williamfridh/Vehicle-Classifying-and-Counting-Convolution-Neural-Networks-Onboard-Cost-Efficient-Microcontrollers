#ifndef MAKE_MFCC_H
#define MAKE_MFCC_H

#include <string>
#include <vector>

// Function prototypes
void makeMfcc(std::vector<std::vector<float>>& curMfcc, const std::vector<float>& x, int sr, int num_mfcc, int num_mel);

#endif // MAKE-MFCC_H