
SHELL	= /bin/sh
CC	= g++3
MACROS  =
CFLAGS	= -Wall -O3

all: Random.o Pattern.o Data.o PatternList.o

Random.o: Random.h Random.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c Random.C

Pattern.o: ../templates/Exceptions.o Pattern.h Pattern.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c Pattern.C

Data.o: ../templates/Exceptions.o ../templates/DynamicArray.h ../templates/DynamicArray.C Pattern.o Data.h Data.C Makefile 
	$(CC) $(CFLAGS) $(MACROS) -c Data.C

PatternList.o: ../templates/Exceptions.o Pattern.o PatternList.h PatternList.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c PatternList.C

install: all

clean:
	rm -f *~ *.o core


