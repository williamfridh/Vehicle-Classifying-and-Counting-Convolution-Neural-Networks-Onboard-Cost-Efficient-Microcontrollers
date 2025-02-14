CC = g++
CFLAGS = -Wall -g -lsndfile -lsamplerate -I./ -O2

TARGET = dataset-converter
SRC = dataset-converter.cc audio-processing.cc make-mfcc.cpp wavreader.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC) $(CFLAGS)

clean:
	rm -f $(TARGET)
