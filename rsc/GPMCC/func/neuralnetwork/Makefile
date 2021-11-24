
SHELL	= /bin/sh
CC	= g++
MACROS  =
CFLAGS  = -Wall -O3

all: NeuralNetwork.o

NeuralNetwork.o: ../baseClasses/Data.o ../baseClasses/Random.o ../baseClasses/Pattern.o NeuralNetwork.h NeuralNetwork.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c NeuralNetwork.C

install: all

clean:
	rm -f *~ *.o core


