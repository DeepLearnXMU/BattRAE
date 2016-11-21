# Executable
EXE    = battrae-model

# Compiler, Linker Defines
CC      = g++
CFLAGS  = -Wall -O2 -Wno-deprecated -m64 -I. -std=c++11 -fopenmp

# Compile and Assemble C++ Source Files into Object Files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
# Source and Object files
SRC    = $(wildcard *.cpp)
OBJ    = $(patsubst %.cpp, %.o, $(SRC))
	
# Link all Object Files with external Libraries into Binaries
$(EXE): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) liblbfgs.a -o $(EXE) -lz

.PHONY: clean
clean:
	 -rm -f core *.o
