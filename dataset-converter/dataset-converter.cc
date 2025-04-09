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
#include <thread>
#include <mutex>
#include "thread_pool.h"


namespace fs = std::filesystem;

std::mutex fileMutex;  // Global mutex for file access


static const std::string ALLOWED_FILE_EXTENSIONS[] = {".wav", ".mp3", ".flac", ".ogg"};

// Constants for default values
const std::string DEFAULT_SOURCE_PATH = "dataset";
const std::string DEFAULT_OUTPUT_PATH = "output_frames";
const std::string DEFAULT_FILENAME_PATH = ".csv";

const int DEFAULT_TARGET_SAMPLE_RATE = 16000;
const float DEFAULT_PRE_EMPHASIS_ALPHA = 0.97;
const float DEFAULT_FRAME_SECONDS = 0.25;
const float DEFAULT_FRAME_OVERLAP_SECONDS = 0.125;
const float NUMBER_OF_MFCC = 16;
const float NUMBER_OF_MEL_BANDS = 32;


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
 * Get average audio level.
 * 
 * This function calculates the average audio level of a frame.
 * 
 * @param frame: Audio frame
 * @return: Average audio level
 */
float getAverageAudioLevel(const std::vector<float>& frame) {
    float sum = 0;
    for (float sample : frame) {
        sum += std::abs(sample);
    }
    return sum / frame.size();
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
    //SF_INFO sfInfo;
    SF_INFO sfInfo = {0};
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfInfo);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << ": " << sf_strerror(NULL) << std::endl;
        return {};
    }
    if ((sfInfo.format & SF_FORMAT_TYPEMASK) != SF_FORMAT_WAV) {
        std::cerr << "Error: File " << filename << " is not in WAV format." << std::endl;
        sf_close(file);
        return {};
    }

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
    if (!fs::exists(dirPath)) {
        if (fs::create_directory(dirPath)) {
            std::cout << "Directory: " << dirPath << " created successfully!" << std::endl;
        } else {
            std::cerr << "Failed to create directory." << std::endl;
        }
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
std::mutex fileWriteMutex;  // Mutex for file writing
int processFile (std::string filePath, std::string outputPath, std::string filenamePath, int targetSampleRate, float preEmphasisAlpha, float frameSeconds, float frameOverlapSeconds) {
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
    //std::cout << "Processing file: " << filePath << std::endl;
    int sampleRate;
    int channels;
    std::vector<float> audioData = readWavFile(filePath, sampleRate, channels);
    if (audioData.empty()) {
        std::cerr << "Error: Failed to read audio data from file: " << filePath << std::endl;
        return 1;
    }

    // Send for pre-processing
    if (channels > 2) {
        std::cerr << "Error: More than 2 channels are not supported." << std::endl;
        return 1;
    }
    if (channels == 2) {
        audioData = stereoToMono(audioData);
    }
    audioData = resampleAudio(audioData, sampleRate, targetSampleRate, channels);
    // Print first 20 samples of audio data
    //std::cout << "First 20 samples of audio data: ";
    //for (size_t i = 0; i < 20 && i < audioData.size(); ++i) {
    //    std::cout << audioData[i] << " ";
    //}
    //std::cout << std::endl;

    // Generate frames
    std::vector<std::vector<float>> frames = generateFrames(audioData, frameSeconds, frameOverlapSeconds, targetSampleRate);
    
    fs::path outputDir = outputPath;  
    fs::path filePathP = filenamePath;  
    fs::path filePathLabel = filePath;  
    fs::path outputFilePath = outputDir / filePathP;
    
    bool printDebug = false;
    {
        std::lock_guard<std::mutex> lock(fileWriteMutex);  // Lock the mutex for file writing

        // Opens file to write to  
        std::ofstream outFile(outputFilePath, std::ios::app);
        if (!outFile.is_open()) {
            std::cerr << "Error: Could not open file for writing.\n";
            return 1;
        }
        // Write frames to files
        for (size_t i = 0; i < frames.size(); ++i) {
            std::vector<float> frame = frames[i]; 








            // Normalize frame data
            frame = normalizeAudio(frame);
            if (printDebug) {
                // Print first 20 samples of audio data
                std::cout << "First 20 samples of audio data: ";
                for (size_t i = 0; i < 20 && i < frame.size(); ++i) {
                    std::cout << frame[i] << " ";
                }
                std::cout << std::endl;
            }
            frame = rmsNormalize(frame, 0.2);
            if (printDebug) {
                // Print first 20 samples of audio data after normalization
                std::cout << "First 20 samples of audio data after normalization: ";
                for (size_t i = 0; i < 20 && i < frame.size(); ++i) {
                    std::cout << frame[i] << " ";
                }
                std::cout << std::endl;
            }
            frame = preEmphasis(frame, preEmphasisAlpha);
            if (printDebug) {
                // Print first 20 samples of audio data after pre-emphasis
                std::cout << "First 20 samples of audio data after pre-emphasis: ";
                for (size_t i = 0; i < 20 && i < frame.size(); ++i) {
                    std::cout << frame[i] << " ";
                }
                std::cout << std::endl;
            }

            printDebug = false;










            std::vector<std::vector<float>> mfcc_matrix = makeMfcc(frame, targetSampleRate, NUMBER_OF_MFCC, NUMBER_OF_MEL_BANDS);

            // Extract label from parent directory 
            fs::path parentDir = filePathLabel.parent_path();
            std::string label = parentDir.filename().string();
            
            writeMfccToCsv(mfcc_matrix, outFile, label);
        }
        
        outFile.close();
    }


    //std::cout << "Frames generated: " << frames.size() << std::endl;
    return 0;
}

std::mutex coutMutex; // Mutex for thread-safe console output


// Convert float to string with 2 decimal places
std::string floatToString(float value, int precision = 2) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    return ss.str();
}


std::string createCsv(std::string outputPath, std::string filenamePath) {
    // Convert to paths
    std::string frame_seconds = floatToString(DEFAULT_FRAME_SECONDS, 2);
    std::string frame_overlap = floatToString(DEFAULT_FRAME_OVERLAP_SECONDS, 2); 
    std::string num_mfcc = floatToString(NUMBER_OF_MFCC, 0); 

    std::string prefix = "seconds_per_frame:" + frame_seconds + 
                     ",overlap:" + frame_overlap + 
                     ",mfccs:" + num_mfcc + "_";

    std::string newFileNamePath = prefix + filenamePath;
    fs::path outputDir = outputPath;  
    fs::path filePathP = prefix + newFileNamePath;  
    // Create directory if it doesn't exist
    if (!fs::exists(outputDir)) {
        std::cout << "Directory does not exist. Creating: " << outputDir << std::endl;
        if (!fs::create_directories(outputDir)) {
            std::cerr << "Error: Unable to create directory.\n";
        }
    }
    
    // Create output file path
    fs::path outputFilePath = outputDir / filePathP;


    return newFileNamePath;
}



void processFileThread(std::string filePath, std::string outputPath, std::string filenamePath, int targetSampleRate, float preEmphasisAlpha, float frameSeconds, float frameOverlapSeconds) {
    int result = processFile(filePath, outputPath, filenamePath, targetSampleRate, preEmphasisAlpha, frameSeconds, frameOverlapSeconds);
    if (result != 0) {
        std::lock_guard<std::mutex> lock(coutMutex);
        std::cerr << "Error processing file: " << filePath << std::endl;
    }
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
 * @param pool: Reference to the thread pool
 * @return: 0 if successful, 1 if error occurs during processing
 */
int iterateFolder (std::string sourcePath, std::string outputPath, std::string filenamePath, int targetSampleRate, float preEmphasisAlpha, float frameSeconds, float frameOverlapSeconds, ThreadPool& pool) {
    // Check if folder exists
    if (!fs::exists(sourcePath)) {
        std::cerr << "Error: Folder does not exist." << std::endl;
        return 1;
    }

    // Read folder
    for (const auto & entry : fs::directory_iterator(sourcePath)) {
        // Check if it is a file
        if (fs::is_regular_file(entry.path())) {
            // Process file using thread pool
            pool.enqueue(processFileThread, entry.path().string(), outputPath, filenamePath, targetSampleRate, preEmphasisAlpha, frameSeconds, frameOverlapSeconds);
        } else if (fs::is_directory(entry.path())) {
            // Recurse
            iterateFolder(entry.path(), outputPath, filenamePath, targetSampleRate, preEmphasisAlpha, frameSeconds, frameOverlapSeconds, pool);
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
    std::cout << "Enter the source folder path (default is '" << DEFAULT_SOURCE_PATH << "'): ";
    std::getline(std::cin, sourcePath);
    if (sourcePath.empty()) sourcePath = DEFAULT_SOURCE_PATH;

    std::string outputPath;
    std::cout << "Enter the output folder path (default is '" << DEFAULT_OUTPUT_PATH << "'): ";
    std::getline(std::cin, outputPath);
    if (outputPath.empty()) outputPath = DEFAULT_OUTPUT_PATH;

    std::string filenamePath;
    std::cout << "Enter name for csv file (default is '" << DEFAULT_FILENAME_PATH << "'): ";
    std::getline(std::cin, filenamePath);
    if (filenamePath.empty()) filenamePath = DEFAULT_FILENAME_PATH;

    std::string targetSampleRate;
    std::cout << "Target sample rate (default is " << DEFAULT_TARGET_SAMPLE_RATE << "): ";
    std::getline(std::cin, targetSampleRate);
    if (targetSampleRate.empty()) targetSampleRate = std::to_string(DEFAULT_TARGET_SAMPLE_RATE);

    std::string preEmphasisAlpha;
    std::cout << "Pre-emphasis alpha (default is " << DEFAULT_PRE_EMPHASIS_ALPHA << "): ";
    std::getline(std::cin, preEmphasisAlpha);
    if (preEmphasisAlpha.empty()) preEmphasisAlpha = std::to_string(DEFAULT_PRE_EMPHASIS_ALPHA);

    std::string frameSeconds;
    std::cout << "Frame seconds (default is " << DEFAULT_FRAME_SECONDS << "): ";
    std::getline(std::cin, frameSeconds);
    if (frameSeconds.empty()) frameSeconds = std::to_string(DEFAULT_FRAME_SECONDS);

    std::string frameOverlapSeconds;
    std::cout << "Frame overlap seconds (default is " << DEFAULT_FRAME_OVERLAP_SECONDS << "): ";
    std::getline(std::cin, frameOverlapSeconds);
    if (frameOverlapSeconds.empty()) frameOverlapSeconds = std::to_string(DEFAULT_FRAME_OVERLAP_SECONDS);

    // Create csv file with headers
    std::string newfileNamePath = createCsv(outputPath, filenamePath);

    // Create thread pool with a number of threads equal to the hardware concurrency
    ThreadPool pool(std::thread::hardware_concurrency());

    // Iterate through the folder
    iterateFolder(sourcePath, outputPath, newfileNamePath, std::stoi(targetSampleRate), std::stof(preEmphasisAlpha), std::stof(frameSeconds), std::stof(frameOverlapSeconds), pool);


    return 0;
}

