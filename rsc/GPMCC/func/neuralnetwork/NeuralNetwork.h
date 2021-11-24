#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "../baseClasses/Random.h"
#include "../baseClasses/Data.h"
#include "../k-means/Clusterer.h"

class NeuralNetwork {
 public:
  NeuralNetwork(Data *data, Clusterer *);
  ~NeuralNetwork();
  void Optimize(double *, char *);

 private:
  void formatPattern(Pattern &, const char *, double *);
  void forwardPropagate(double *, double *, double *);
  void clearDeltas();
  double backPropagate(const double *, const double *, const double *);
  double calculateSSE(const double *, const double *, const double *);
  void updateWeights();
  void pickPatterns();
  void shuffle();
  void trainingPatterns();
  void validationPatterns();
  void generalizationPatterns();

  static Data *data;
  unsigned noInputs;
  unsigned noOutputs;
  unsigned noHiddens;
  double lambda;
  double eta;
  double alpha;
  unsigned epochs;
  double **inputHidden;
  double **hiddenOutput;

  double **inputHidden_dw;
  double **hiddenOutput_dw;
  double **inputHidden_dwprev;
  double **hiddenOutput_dwprev;

  unsigned sampleSize;
  unsigned noTrainingPatterns;
  Clusterer *strata;
  DynamicArray<Pattern *> *sample;
};

#endif
