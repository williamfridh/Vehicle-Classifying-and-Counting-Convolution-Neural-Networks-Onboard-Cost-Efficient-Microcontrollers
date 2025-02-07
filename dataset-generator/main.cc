/**
 * Dataset Generator.
 * 
 * This dataset generator is used for reading audio files and extracting features from them.
 * It takes in a folder path containing nested folders and audio files. Then it reads each
 * audio file and copies a segment of it into a new file. It first takes the first X seconds
 * then travels Y seconds and takes the next X seconds and so on. This is done to create
 * a dataset of overlaping segments of audio files.
 * 
 */

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * Iterate Folder.
 * 
 * This function takes in a folder path and iterates through all the files and folders
 * in it. During each iteration it passes the path of each file to another function
 * that processes the file.
 */
int iterateFolder (std::string folderPath) {
    // Check if folder exists
    if (!fs::exists(folderPath)) {
        std::cerr << "Error: Folder does not exist." << std::endl;
        return 1;
    }
    // Read folder
    for (const auto & entry : fs::directory_iterator(folderPath)) {
        std::cout << entry.path() << std::endl;
    }
    // Return
    return 0;
}


/**
 * Main Function.
 */
int main (int argc, char *argv[]) {
    // Take console input
    std::string folderPath;
    std::cout << "Enter the folder path: ";
    std::cin >> folderPath;
    // Itterate through the folder
    iterateFolder(folderPath);
    // Return
    return 0;
}

