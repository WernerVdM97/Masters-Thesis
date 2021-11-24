#ifndef CHROMOPOOL_H
#define CHROMOPOOL_H

#include <iostream>
#include "../templates/DynamicArray.h"
#include "../baseClasses/Data.h"
#include "../baseClasses/Random.h"
#include "../k-means/Clusterer.h"
#include "../gapolycurve/Chromo.h"
#include "../gapolycurve/GAPolyCurve.h"

struct PoolData {
  Chromo *chromo;
  unsigned usage;
  unsigned sinceUsed;
};

class ChromoPool {
 public:
  ChromoPool(Data *);
  ~ChromoPool();

  Chromo *Get();
  Chromo *Add(Chromo *, bool);
  void Remove(Chromo *);
  void Maintenance();
  Pattern * GetPattern();
  void Resample();
  DynamicArray<Pattern *>* GetTrainingSet();
  DynamicArray<Pattern *>* GetSample();

 private:
  static Data *data;
  DynamicArray<PoolData *> *pool;
  DynamicArray<Pattern *> *strata;
  unsigned noStrata;
  DynamicArray<Pattern *> *sample;
  DynamicArray<Pattern *> *overall;
  unsigned noTrainingSet;
  double sampleSize;
  bool gaussian;

  void InitialSample();

  void Sort(DynamicArray<PoolData *> &);
  void QSort(DynamicArray<PoolData *> &, int, int);
};

#endif
