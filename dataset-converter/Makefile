CC = g++
CFLAGS = -Wall -g -lsndfile -lsamplerate
TARGET = dataset-converter
SRC = dataset-converter.cc audio-processing.cc

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC) $(CFLAGS)

clean:
	rm -f $(TARGET)