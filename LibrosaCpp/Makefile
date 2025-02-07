# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -I./ -O2

# Target executable
TARGET = testCompiled

# Source files
SRCS = test/test.cc test/wavreader.c

# Object files
OBJS = $(SRCS:.cc=.o)
OBJS := $(OBJS:.c=.o)

# Default target
all: $(TARGET)

# Link the target executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

# Compile source files into object files
test/%.o: test/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

test/%.o: test/%.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean