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

#include "main_functions.h"
#include "pico/stdlib.h"
#include <cstdio>

// This is the default main used on systems that have the standard C entry
// point. Other devices (for example FreeRTOS or ESP32) that have different
// requirements for entry code (like an app_main function) should specialize
// this main.cc file in a target-specific subfolder.
int main(int argc, char* argv[]) {
  // Initialize serial communication.
  stdio_init_all();
  // Artificial delay to prevent missing first serial output(s).
  sleep_ms(START_DELAY);
  // Setup program.
  setup(5000);
  // Check for setup errors.
  if (setupError) {
    printf("Setup failed\n");
    return 1;
  }
  // Everything looks green! Ready to start.
  printf("Ready to start\n");
  while (true) {
    loop();
  }
}
