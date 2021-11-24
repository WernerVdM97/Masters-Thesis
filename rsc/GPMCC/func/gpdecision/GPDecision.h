#ifndef GPDECISION_H
#define GPDECISION_H

#include "ChromoPool.h"
#include "DecisionTree.h"

class GPDecision {
 public:
  GPDecision(Data *);
  ~GPDecision();

  DecisionTree* Optimize();

 private:
  Data *data;
  DecisionTree *best;
  DecisionTree **individuals;
  unsigned noIndividuals;
  ChromoPool *pool;

  void Sort(DecisionTree **, unsigned);
  void QSort(DecisionTree **, int, int);

  double CalculateSampleStd(DynamicArray<Pattern *> *);
  double tstd;
  double vstd;
};

#endif
