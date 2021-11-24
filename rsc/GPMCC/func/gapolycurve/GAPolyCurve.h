#ifndef GA_POLY_CURVE_H
#define GA_POLY_CURVE_H

#include "../templates/DynamicArray.h"
#include "../baseClasses/Random.h"
#include "../baseClasses/Pattern.h"
#include "../baseClasses/PatternList.h"
#include "../k-means/Cluster.h"
#include "../k-means/Clusterer.h"

#include "Chromo.h"

class GAPolyCurve {
 public:
  GAPolyCurve(Data *, Clusterer *, DynamicArray<Pattern *> *, Chromo *);
  ~GAPolyCurve();

  Chromo& Optimize(
#if SHOW_GAPOLYCURVE == 1
double * outArray
#endif
);

 private:
  Chromo **individuals;
  Chromo **highScores;
  Data *data;
  Clusterer *strata;
  DynamicArray<Pattern *> *sample;
  DynamicArray<Pattern *> *validationSet;
  unsigned classification;

  void Sort(Chromo **, unsigned);
  void QSort(Chromo **, int, int);
  void ManageHighScores(Chromo *chromo);

  void PickPatterns();
  void TrainingPatterns();
  void ValidationPatterns();
  void GeneralizationPatterns();
  void SampleStd();

  unsigned noTrainingPatterns;
  unsigned sampleSize;
  double std;
  unsigned maximumComponents;

};

#endif
