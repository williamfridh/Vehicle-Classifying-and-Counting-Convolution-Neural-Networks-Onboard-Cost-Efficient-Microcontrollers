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
#include <cmath>
#include <sndfile.h>
#include <samplerate.h>

namespace fs = std::filesystem;

static const std::string ALLOWED_FILE_EXTENSIONS[] = {".wav", ".mp3", ".flac", ".ogg"};

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
std::vector<float> resampleAudio(const std::vector<float>& inputAudio, int inputSampleRate, int targetSampleRate) {
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
    srcData.output_frames = outputSize;
    srcData.src_ratio = ratio;
    srcData.end_of_input = 0;

    if (src_simple(&srcData, SRC_SINC_BEST_QUALITY, 1) != 0) {
        std::cerr << "Error: Resampling failed!" << std::endl;
        return {};
    }

    return outputAudio;
}


/**
 * Compute RMS.
 * 
 * This function computes the Root Mean Square (RMS) of the audio data.
 * 
 * @param audio: Audio data
 * @return: RMS value of the audio data
 */
float computeRMS(const std::vector<float>& audio) {
    float sum = 0.0;
    for (float sample : audio) {
        sum += sample * sample;
    }
    return sqrt(sum / audio.size());
}

/**
 * RMS Normalize.
 * 
 * This function normalizes the audio data to a target RMS value.
 * 
 * @param audio: Audio data
 * @param targetRMS: Target RMS value (default: 0.1)
 * @return: Normalized audio data
 */
std::vector<float> rmsNormalize(const std::vector<float>& audio, float targetRMS = 0.1) {

    if (targetRMS < 0.1 || targetRMS > 0.3) {
        std::cerr << "Warning: Target RMS value should be between 0.1 and 0.3" << std::endl;
    }

    float currentRMS = computeRMS(audio);
    
    // Prevent division by zero
    if (currentRMS < 1e-8) {
        return audio; // Return original audio if the RMS is too small
    }

    float gain = targetRMS / currentRMS;
    
    std::vector<float> normalizedAudio(audio.size());
    for (size_t i = 0; i < audio.size(); i++) {
        normalizedAudio[i] = audio[i] * gain;
    }
    
    return normalizedAudio;
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
std::vector<float> readWavFile(const std::string& filename, int& sampleRate) {
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
 * Process single file.
 * 
 * This function takes a file path and processes the file. It reads the file, checks
 * the file extension, sends it for pre-reprocessing(normalization, trimming, etc),
 * and then performs slicing to create the segements of the file.
 * 
 * @param filePath: Path of the file to be processed
 * @param outputPath: Path of the output folder where the processed files will be stored
 * @param targetSampleRate: Target sample rate for resampling
 * @return: 0 if successful, 1 if error occurs during processing
 */
int processFile (std::string filePath, std::string outputPath, int targetSampleRate) {
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
    std::vector<float> audioData = readWavFile(filePath, sampleRate);
    // Send for pre-processing
    audioData = rmsNormalize(audioData, 0.2);
    audioData = resampleAudio(audioData, sampleRate, targetSampleRate);
    // Write file at new path
    std::string newFilePath = outputPath + "/" + filePath;
    writeWavFile(newFilePath, audioData, sampleRate, 1);
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
 * @return: 0 if successful, 1 if error occurs during processing
 */
int iterateFolder (std::string sourcePath, std::string outputPath, int targetSampleRate) {
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
            processFile(entry.path(), outputPath, targetSampleRate);
        } else if (fs::is_directory(entry.path())) {
            // Recurse
            iterateFolder(entry.path(), outputPath, targetSampleRate);
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
    std::cout << "Enter the output folder path (default is 'processed'): ";
    std::getline(std::cin, outputPath);
    if (outputPath.empty()) outputPath = "processed";

    std::string targetSampleRate;
    std::cout << "Target sample rate (default is '16000'): ";
    std::getline(std::cin, targetSampleRate);
    if (targetSampleRate.empty()) targetSampleRate = "16000";

    // Itterate through the folder
    iterateFolder(sourcePath, outputPath, std::stoi(targetSampleRate));
    // Return
    return 0;
}

