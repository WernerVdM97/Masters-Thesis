#ifndef CLUSTERER_H
#define CLUSTERER_H

#include <iostream>
#include "../templates/Exceptions.h"
#include "../templates/DynamicArray.h"
#include "../baseClasses/Data.h"
#include "../baseClasses/Pattern.h"
#include "Cluster.h"

class Clusterer {
 public:
  Clusterer(Data *, DynamicArray<Pattern *> *, DynamicArray<Pattern *> *, unsigned, unsigned);
  ~Clusterer();

  void NoHeuristicOptimize();
  void FirstHeuristicOptimize();
  void SecondHeuristicOptimize();
  void Finalize();
  unsigned Length() const;
  Cluster& operator [] (unsigned) const;

 private:
  Data *data;
  Cluster *clusters;
  unsigned noClusters;
  unsigned noEpochs;
};

#endif
