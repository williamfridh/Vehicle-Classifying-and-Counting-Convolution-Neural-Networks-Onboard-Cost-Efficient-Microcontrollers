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

static const std::string ALLOWED_FILE_EXTENSIONS[] = {".wav", ".mp3", ".flac", ".ogg"};

/**
 * Normalize audio.
 * 
 * Lightweight audio pre-processing function that normalizes the audio.
 * It takes in the audio data and normalizes it to a certain level.
 */
int normalizeAudio (std::string filePath) {
    // Read audio file
    
}

/**
 * Process single file.
 * 
 * This function takes a file path and processes the file. It reads the file, checks
 * the file extension, sends it for pre-reprocessing(normalization, trimming, etc),
 * and then performs slicing to create the segements of the file.
 */
int processFile (std::string filePath) {
    // Note that no checking for file existence is done here
    // as it is assumed that the file exists.

    // Check if file is allowed
    std::string fileExtension = fs::path(filePath).extension();
    bool isAllowed = false;
    for (std::string allowedExtension : ALLOWED_FILE_EXTENSIONS) {
        if (fileExtension == allowedExtension) {
            isAllowed = true;
            break;
        }
    }
    if (!isAllowed) {
        std::cerr << "Error: File type not allowed. File path: " << filePath << std::endl;
        return 1;
    }
    // Read file
    std::cout << "Processing file: " << filePath << std::endl;
    // Return
    return 0;
}

/**
 * Iterate Folder.
 * 
 * This function takes in a folder path and iterates through all the files and folders
 * in it. During each iteration it passes the path of each file to another function
 * that processes the file.
 */
int iterateFolder (std::string sourcePath, std::string outputPath) {
    // Check if folder exists
    if (!fs::exists(sourcePath)) {
        std::cerr << "Error: Folder does not exist." << std::endl;
        return 1;
    }
    // Read folder
    for (const auto & entry : fs::directory_iterator(sourcePath)) {
        // Check if it is a file
        if (fs::is_regular_file(entry.path())) {
            // Process file
            processFile(entry.path());
        } else if (fs::is_directory(entry.path())) {
            // Recurse
            iterateFolder(entry.path(), outputPath);
        }
    }
    // Return
    return 0;
}

/**
 * Main Function.
 */
int main (int argc, char *argv[]) {
    // Take console input
    std::string sourcePath;
    std::cout << "Enter the source folder path: ";
    std::cin >> sourcePath;

    std::string outputPath;
    std::cout << "Enter the output folder path: ";
    std::cin >> outputPath;

    // Itterate through the folder
    iterateFolder(sourcePath, outputPath);
    // Return
    return 0;
}

