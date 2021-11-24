
SHELL	= /bin/sh
CC	= g++3
MACROS	=
CFLAGS	= -Wall -O3

all: Cluster.o Clusterer.o

Cluster.o: ../baseClasses/Pattern.o ../baseClasses/PatternList.o Cluster.h Cluster.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c Cluster.C

Clusterer.o: ../templates/DynamicArray.h ../templates/DynamicArray.C ../templates/Exceptions.o ../baseClasses/Data.o ../baseClasses/Pattern.o Cluster.o Clusterer.h Clusterer.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c Clusterer.C

install: all

clean:
	rm -f *~ *.o core




