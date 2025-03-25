/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "constants.h"

// This is a small number so that it's easy to read the logs
const int kInferencesPerCycle = 20;

#define NUM_OF_CLASSES 4                                                        // Number of classes.
#define MAX_NEGATIVE_GUESSES 2
#define NUM_OF_MFCC 400
#define NEGATIVE_CLASS_INDEX 0                                                  // Index of the negative class.

const char* CLASSES[NUM_OF_CLASSES]= {"Car", "Truck", "Bus", "Motorcycle"};     // Possible classes (labels).