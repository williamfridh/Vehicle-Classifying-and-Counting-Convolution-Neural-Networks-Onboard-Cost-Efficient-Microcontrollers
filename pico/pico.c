/*
* This file contains the foundation for classifing audio data in the form of MFCCs.
* 
* TODO:
* 1. Finalize the input array. Ones it's filled up, the classification should happen. Once the classification is done, the array should be reset.
* 2. Implement the actual model.
*/


#include <stdio.h>
#include <stdlib.h>
#include "pico/stdlib.h"

#define NUM_OF_CLASSES 4                                                        // Number of classes.
#define MAX_NEGATIVE_GUESSES 2
#define NUM_OF_MFCC 400
#define NEGATIVE_CLASS_INDEX 0                                                  // Index of the negative class.

const char* CLASSES[NUM_OF_CLASSES]= {"Car", "Truck", "Bus", "Motorcycle"};     // Possible classes (labels).

int call_model(float* input)
{
    // Call the model and return the classification.
    // This is just a placeholder for the actual model.
    return rand() % NUM_OF_CLASSES;
}

void classify(float* input, int* guesses, int* background_noise, int* classifications)
{
    int guess = call_model(input);
    if (guess == NEGATIVE_CLASS_INDEX) {
        (*background_noise)++;      // Increment background noise counter.
    } else {
        *background_noise = 0;      // Reset background noise counter.
        guesses[guess]++;    // Increment the guess counter.
    }

    // If background noise is detected for X consecutive frames,
    // find the class with the highest number of guesses and classify it.
    if (*background_noise >= MAX_NEGATIVE_GUESSES) {
        int max = 0;
        int max_index = 0;
        for (int i = 0; i < NUM_OF_CLASSES; i++) {
            if (guesses[i] > max) {
                max = guesses[i];
                max_index = i;
            }
        }
        classifications[max_index]++;
        // Reset the guesses.
        for (int i = 0; i < NUM_OF_CLASSES; i++) {
            guesses[i] = 0;
        }
        // Reset the background noise counter.
        *background_noise = 0;
    }
}

void print_status(int* guesses, int* background_noise, int* classifications)
{

    printf("Guesses: ");
    for (int i = 0; i < NUM_OF_CLASSES; i++) {
        printf("%s: %d, ", CLASSES[i], guesses[i]);
    }
    printf("\n");

    printf("Background noise: %d\n", *background_noise);

    printf("Classifications: ");
    for (int i = 0; i < NUM_OF_CLASSES; i++) {
        printf("%s: %d, ", CLASSES[i], classifications[i]);
    }
    printf("\n");
}

int main()
{
    stdio_init_all();

    float   input[NUM_OF_MFCC] = {0};                   // Flattended MFCC array.
    int     guesses[NUM_OF_CLASSES] = {0};              // Possible classes.
    int     background_noise = 0;                       // Number of background noise classifications.
    int     classifications[NUM_OF_CLASSES] = {0};      // Number of classifications.

    while (true) {
        classify(input, guesses, &background_noise, classifications);
        print_status(guesses, &background_noise, classifications);
        sleep_ms(1000);
    }
}
