
SHELL	= /bin/sh
CC	= g++3
MACROS  =
CFLAGS	= -Wall -O3

all: Exceptions.o Matrix.o

Exceptions.o: Exceptions.h Exceptions.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c Exceptions.C

Matrix.o: Exceptions.o Matrix.h Matrix.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c Matrix.C

install: all

clean:
	rm -f *~ *.o core


