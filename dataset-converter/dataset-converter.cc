/**
 * Dataset Generator.
 * 
 * This dataset generator is used for reading audio files and extracting features from them.
 * It takes in a folder path containing nested folders and audio files. Then it reads each
 * audio file and copies a segment of it into a new file. It first takes the first X seconds
 * then travels Y seconds and takes the next X seconds and so on. This is done to create
 * a dataset of overlaping segments of audio files.
 * 
 * Audio processing is done using the libsndfile and libsamplerate libraries.
 * 
 * This code does the following to audio files:
 * 1. Convert into mono channel.
 * 2. Resamples the audio to a target sample rate.
 * 3. Normalizes the audio to a target RMS value.
 * 4. Applies pre-emphasis to the audio.
 * 
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <cmath>
#include <sndfile.h>
#include <samplerate.h>
#include "audio-processing.h"
#include "make-mfcc.h"


namespace fs = std::filesystem;

static const std::string ALLOWED_FILE_EXTENSIONS[] = {".wav", ".mp3", ".flac", ".ogg"};

/**
 * Convert stereo audio to mono.
 * 
 * This function converts stereo audio to mono by averaging the left and right channels.
 * 
 * @param audio: Stereo audio data
 * @return: Mono audio data
 */
std::vector<float> stereoToMono(const std::vector<float>& audio) {
    if (audio.size() % 2 != 0) {
        std::cerr << "Warning: Stereo audio size is not even." << std::endl;
    }

    std::vector<float> mono(audio.size() / 2);
    for (size_t i = 0; i < audio.size(); i += 2) {
        mono[i / 2] = (audio[i] + audio[i + 1]) / 2;
    }

    return mono;
}

/**
 * Generate frames.
 * 
 * This function takes audio data and frames it into segments of a given size,
 * with a given overlap. The frame size and overlap are specified in seconds.
 * 
 * @param audio: Audio data
 * @param frameSeconds: Size of each frame in seconds
 * @param frameOverlapSeconds: Overlap between frames in seconds
 * @param sampleRate: Sample rate of the audio
 * @return: Vector of framed audio segments
 */
std::vector<std::vector<float>> generateFrames(const std::vector<float>& audio, float frameSeconds, float frameOverlapSeconds, int sampleRate) {
    std::vector<std::vector<float>> frames;

    // Convert frame size and overlap from seconds to samples
    int frameLength = static_cast<int>(frameSeconds * sampleRate);
    int overlapLength = static_cast<int>(frameOverlapSeconds * sampleRate);

    // Ensure frameLength is greater than overlapLength to avoid infinite loop
    if (frameLength <= overlapLength) {
        throw std::invalid_argument("Frame length (" + std::to_string(frameLength) + ") must be greater than overlap length (" + std::to_string(overlapLength) + ") at sample rate " + std::to_string(sampleRate) + ".");
    }

    // Step through signal with stride (frameLength - overlapLength)
    for (size_t i = 0; i + frameLength <= audio.size(); i += (frameLength - overlapLength)) {
        // Extract frame from signal
        std::vector<float> frame(audio.begin() + i, audio.begin() + i + frameLength);
        frames.push_back(frame);
    }

    return frames;
}

/**
 * Resample Audio.
 * 
 * This function resamples the audio data to a target sample rate.
 * 
 * @param inputAudio: Input audio data
 * @param inputSampleRate: Sample rate of the input audio
 * @param targetSampleRate: Target sample rate
 * @return: Resampled audio data
 */
std::vector<float> resampleAudio(const std::vector<float>& inputAudio, int inputSampleRate, int targetSampleRate, int channels) {
    if (inputSampleRate == targetSampleRate) return inputAudio;  // No resampling needed
    if (!targetSampleRate) {
        std::cerr << "Error: Target sample rate cannot be zero!" << std::endl;
        return {};
    }

    float ratio = (float)targetSampleRate / inputSampleRate;
    int outputSize = static_cast<int>(inputAudio.size() * ratio);

    std::vector<float> outputAudio(outputSize);

    SRC_DATA srcData;
    srcData.data_in = inputAudio.data();
    srcData.input_frames = inputAudio.size();
    srcData.data_out = outputAudio.data();
    srcData.output_frames = outputSize / channels;
    srcData.src_ratio = ratio;
    srcData.end_of_input = 0;

    if (src_simple(&srcData, SRC_SINC_BEST_QUALITY, 1) != 0) {
        std::cerr << "Error: Resampling failed!" << std::endl;
        return {};
    }

    return outputAudio;
}

/**
 * Read WAV File.
 * 
 * This function reads a WAV file and returns the audio data along with the sample rate.
 * 
 * @param filename: Path of the WAV file
 * @param sampleRate: Sample rate of the audio
 * @return: Audio data of the WAV file as a vector of floats
 */
std::vector<float> readWavFile(const std::string& filename, int& sampleRate, int &channels) {
    SF_INFO sfInfo;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfInfo);

    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    std::vector<float> audioData(sfInfo.frames * sfInfo.channels);
    sf_read_float(file, audioData.data(), audioData.size());
    sf_close(file);

    sampleRate = sfInfo.samplerate;
    channels = sfInfo.channels;
    return audioData;
}

/**
 * Write WAV File.
 * 
 * This function writes audio data to a WAV file.
 * 
 * @param filename: Path of the WAV file to be written
 * @param audio: Audio data to be written
 * @param sampleRate: Sample rate of the audio
 * @param channels: Number of channels in the audio
 * @return: None
 */
void writeWavFile(const std::string& filename, const std::vector<float>& audio, int sampleRate, int channels) {
    // Create directories if they do not exist
    fs::path filePath = filename;
    fs::create_directories(filePath.parent_path());

    SF_INFO sfInfo;
    sfInfo.samplerate = sampleRate;
    sfInfo.channels = channels;
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* file = sf_open(filename.c_str(), SFM_WRITE, &sfInfo);
    if (!file) {
        std::cerr << "Error: Could not write file " << filename << std::endl;
        return;
    }

    sf_write_float(file, audio.data(), audio.size());
    sf_close(file);
}



/** 
 * Creates a directory to host the txt frames
 *
 * @return: The name of the directory folder    
*/

void makeFrameDirectory(std::string dirName, std::string outerDir) {
    // Making the path to which the txt file will be saved
    // dirName in this case is the name of the directory where the data is gathered 
    // Directory path is supposed to look like "resulting_frames/dirName"

    fs::path outerDirPath = fs::path(outerDir);
    fs::path dirPath = outerDirPath / dirName;

    // Check if outer directory exists, if not create it
    if (!fs::exists(outerDirPath)) {
        if (fs::create_directory(outerDirPath)) {
            std::cout << "Outer directory created successfully!" << std::endl;
        } else {
            std::cerr << "Failed to create outer directory." << std::endl;
            return;
        }
    }

    // Check if inner directory exists, if not create it
    if (fs::exists(dirPath)) {
        std::cout << "Directory already exists." << std::endl;
    } else {
        if (fs::create_directory(dirPath)) {
            std::cout << "Directory created successfully!" << std::endl;
        } else {
            std::cerr << "Failed to create directory." << std::endl;
        }
    }
}

/**
 * Moves the frame txt files.
 *
 * @param txtFilePath: The file path in which to move the file
 * @param destinationDir: The directory in which to move the file 
 * @return: None
*/
void moveTxtFile(fs::path txtFilePath, fs::path destinationDir) {
    fs::path sourcePath(txtFilePath);
    fs::path destinationPath = fs::path(destinationDir) / destinationDir /sourcePath.filename();
    try {
        if (fs::exists(sourcePath)) {
            fs::rename(sourcePath, destinationPath); // Move the file
            std::cout << "File moved successfully to: " << destinationPath << std::endl;
        } else {
            std::cerr << "Source file does not exist: " << sourcePath << std::endl;
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error moving file: " << e.what() << std::endl;
    }
}





/**
 * Process single file.
 * 
 * This function takes a file path and processes the file. It reads the file, checks
 * the file extension, sends it for pre-reprocessing(normalization, trimming, etc),
 * and then performs slicing to create the segements of the file.
 * 
 * @param filePath: Path of the file to be processed
 * @param outputPath: Path of the output folder where the processed files will be stored
 * @param targetSampleRate: Target sample rate for resampling
 * @param preEmphasisAlpha: Pre-emphasis coefficient
 * @param frameSeconds: Size of each frame
 * @param frameOverlapSeconds: Overlap between frames
 * @return: 0 if successful, 1 if error occurs during processing
 */
int processFile (std::string filePath, std::string outputPath, int targetSampleRate, float preEmphasisAlpha, float frameSeconds, float frameOverlapSeconds) {
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
    int sampleRate;
    int channels;
    std::vector<float> audioData = readWavFile(filePath, sampleRate, channels);
    // Send for pre-processing
    audioData = stereoToMono(audioData);
    audioData = resampleAudio(audioData, sampleRate, targetSampleRate, channels);
    audioData = rmsNormalize(audioData, 0.2);
    audioData = preEmphasis(audioData, preEmphasisAlpha);
    // Generate frames
    std::vector<std::vector<float>> frames = generateFrames(audioData, frameSeconds, frameOverlapSeconds, targetSampleRate);
    // Make a new directory and Write new files
    std::string outerFrameTxtDirectory = "resulting_frames";
    for (size_t i = 0; i < frames.size(); ++i) {
        // Process each frame into a mfcc
        std::vector<float> frame = frames[i]; 
        std::string mfccString = makeMfcc(frame, sampleRate);
        // Create name of the txt file, directory, and create the txt file
        
        fs::path pathObj(filePath);
        std::string classifiName = "";

        // Gets the name of the "classification name, or rather dataset/"secondDir"
        auto it = pathObj.begin();
        if (std::distance(pathObj.begin(), pathObj.end()) >= 2) {
            std::advance(it, 1);  // Move iterator to the second directory
            classifiName = it->string();
            std::cout << "Second directory: " << classifiName << std::endl;
        } else {
            std::cout << "Path does not contain enough directories." << std::endl;
        }

        // Creates directory of classification name inside "resulting_frames" directory 
        makeFrameDirectory(classifiName, outputPath);
        
        fs::path outerDir = outputPath;  // Convert string to fs::path
        fs::path classifiDir = outerDir / classifiName;  // Create subdirectory path

        // Created txt file
        fs::path audioFilePath(filePath);
        std::string audioFileName = audioFilePath.stem().string();
        fs::path txtFilePath = classifiDir / (audioFileName + "_frame_number:" + std::to_string(i) + ".txt");
        

        std::cout << "classifiName: " << classifiName << std::endl;
        std::cout << "outerDir: " << outerDir << std::endl;
        std::cout << "classifiDir: " << classifiDir << std::endl;
        std::cout << "txtFilePath: " << txtFilePath << std::endl;

        // "resulting_frames/filePath/audioFileName/_frame_number:i.txt"
        
        // Save data to the file 
        std::ofstream outFile(txtFilePath);
        if (outFile.is_open()) {
            outFile << mfccString;
            outFile.close();
            std::cout << "Saved: " << filePath << std::endl;
            moveTxtFile(txtFilePath, classifiDir);
        } else {
            std::cerr << "Error opening file: " << filePath << std::endl;
        }

        //std::string newFilePath = outputPath + "/" + filePath + "_" + std::to_string(i) + ".wav";
        //writeWavFile(newFilePath, frames[i], targetSampleRate, 1);
    }
    // Print out amount of frames generated
    std::cout << "Frames generated: " << frames.size() << std::endl;
    // Return
    return 0;
}

/**
 * Iterate Folder.
 * 
 * This function takes in a folder path and iterates through all the files and folders
 * in it. During each iteration it passes the path of each file to another function
 * that processes the file.
 * 
 * @param sourcePath: Path of the folder to be iterated
 * @param outputPath: Path of the output folder where the processed files will be stored
 * @param targetSampleRate: Target sample rate for resampling
 * @param preEmphasisAlpha: Pre-emphasis coefficient
 * @param frameSeconds: Size of each frame
 * @param overlap: Overlap between frames
 * @return: 0 if successful, 1 if error occurs during processing
 */
int iterateFolder (std::string sourcePath, std::string outputPath, int targetSampleRate, float preEmphasisAlpha, float frameSeconds, float frameOverlapSeconds) {
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
            processFile(entry.path(), "output_frames", targetSampleRate, preEmphasisAlpha, frameSeconds, frameOverlapSeconds);
        } else if (fs::is_directory(entry.path())) {
            // Recurse
            iterateFolder(entry.path(), "output_frames", targetSampleRate, preEmphasisAlpha, frameSeconds, frameOverlapSeconds);
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
    std::cout << "Enter the source folder path (default is 'dataset'): ";
    std::getline(std::cin, sourcePath);
    if (sourcePath.empty()) sourcePath = "dataset";

    std::string outputPath;
    std::cout << "Enter the output folder path (default is 'output_frames'): ";
    std::getline(std::cin, outputPath);
    if (outputPath.empty()) outputPath = "output_frames";

    std::string targetSampleRate;
    std::cout << "Target sample rate (default is 16000): ";
    std::getline(std::cin, targetSampleRate);
    if (targetSampleRate.empty()) targetSampleRate = "16000";

    std::string preEmphasisAlpha;
    std::cout << "Pre-emphasis alpha (default is 0.97): ";
    std::getline(std::cin, preEmphasisAlpha);
    if (preEmphasisAlpha.empty()) preEmphasisAlpha = "0.97";

    std::string frameSeconds;
    std::cout << "Frame seconds (default is 0.5): ";
    std::getline(std::cin, frameSeconds);
    if (frameSeconds.empty()) frameSeconds = "0.5";

    std::string frameOverlapSeconds;
    std::cout << "Frame overlap seconds (default is 0.25): ";
    std::getline(std::cin, frameOverlapSeconds);
    if (frameOverlapSeconds.empty()) frameOverlapSeconds = "0.25";

    // Itterate through the folder
    iterateFolder(sourcePath, outputPath, std::stoi(targetSampleRate), std::stof(preEmphasisAlpha), std::stof(frameSeconds), std::stof(frameOverlapSeconds));
    // Return
    return 0;
}

