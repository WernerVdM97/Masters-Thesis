
SHELL	= /bin/sh
CC	= g++3
MACROS  =
CFLAGS  = -Wall -O3

all: ChromoPool.o Antecedent.o NominalAntecedent.o ContinuousAntecedent.o Consequent.o ContinuousConsequent.o DecisionNode.o DecisionTree.o GPDecision.o

ChromoPool.o: ../templates/DynamicArray.h ../templates/DynamicArray.C ../baseClasses/Data.o ../baseClasses/Random.o ../baseClasses/Pattern.o ../gapolycurve/Chromo.o ../gapolycurve/GAPolyCurve.o ChromoPool.h ChromoPool.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c ChromoPool.C

Antecedent.o: ../templates/Exceptions.h ../templates/DynamicArray.h ../templates/DynamicArray.C ../baseClasses/Data.o ../baseClasses/Pattern.o Antecedent.h Antecedent.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c Antecedent.C

NominalAntecedent.o: ../baseClasses/Random.o Antecedent.o NominalAntecedent.h NominalAntecedent.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c NominalAntecedent.C

ContinuousAntecedent.o: ../baseClasses/Random.o Antecedent.o ContinuousAntecedent.h ContinuousAntecedent.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c ContinuousAntecedent.C

Consequent.o: ../templates/DynamicArray.h ../templates/DynamicArray.C ../baseClasses/Data.o ../gapolycurve/Chromo.o Consequent.h Consequent.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c Consequent.C

ContinuousConsequent.o: ../k-means/Clusterer.o ../gapolycurve/GAPolyCurve.o Consequent.o ChromoPool.o ContinuousConsequent.h ContinuousConsequent.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c ContinuousConsequent.C

DecisionNode.o: Antecedent.o Consequent.o DecisionNode.h DecisionNode.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c DecisionNode.C

DecisionTree.o: DecisionNode.o NominalAntecedent.o ContinuousAntecedent.o ContinuousConsequent.o DecisionTree.h DecisionTree.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c DecisionTree.C

GPDecision.o: ChromoPool.o DecisionTree.o GPDecision.h GPDecision.C Makefile
	$(CC) $(CFLAGS) $(MACROS) -c GPDecision.C

install: all

clean:
	rm -f *~ *.o core


