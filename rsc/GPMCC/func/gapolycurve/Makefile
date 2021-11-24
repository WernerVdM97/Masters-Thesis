
SHELL	= /bin/sh
CC	= g++3
MACROS	=
CFLAGS	= -Wall -O3

all: Variable.o Component.o Chromo.o GAPolyCurve.o

Variable.o: Variable.h Variable.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c Variable.C

Component.o: ../templates/DynamicArray.C ../templates/DynamicArray.h ../baseClasses/Data.o ../baseClasses/Pattern.o Variable.o Component.h Component.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c Component.C

Chromo.o: ../templates/DynamicArray.C ../templates/DynamicArray.h Component.o ../baseClasses/Data.o ../templates/Matrix.o Chromo.h Chromo.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c Chromo.C

GAPolyCurve.o: ../templates/DynamicArray.h ../templates/DynamicArray.C ../baseClasses/Random.o ../baseClasses/Pattern.o ../baseClasses/PatternList.o Chromo.o GAPolyCurve.h GAPolyCurve.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c GAPolyCurve.C

install: all

clean:
	rm -f *~ *.o core


