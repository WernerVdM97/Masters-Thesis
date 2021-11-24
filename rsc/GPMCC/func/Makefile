
SHELL  = /bin/sh
CC     = g++ -march=i686
CFLAGS = -Wall -O3

export SHELL
export CC
export CFLAGS

all:
	make -f Makefile.gapolycurve -e clean
	make -f Makefile.gapolycurve -e
	make -f Makefile.gpdecision -e clean
	make -f Makefile.gpdecision -e
	make -f Makefile.neuralnetwork -e clean
	make -f Makefile.neuralnetwork -e

install: all

clean:
	cd templates; make clean; cd ..
	cd baseClasses; make clean; cd ..
	cd k-means; make clean; cd ..
	cd gapolycurve; make clean; cd ..
	cd gpdecision; make clean; cd ..
	cd neuralnetwork; make clean; cd ..
	rm -f *~ *.o core gpdecision.exe gapolycurve.exe neuralnetwork.exe
