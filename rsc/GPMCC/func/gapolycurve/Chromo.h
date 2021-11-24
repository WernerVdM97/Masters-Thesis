#ifndef CHROMO_H
#define CHROMO_H

#include <iostream>
#include <string>

#ifdef GCC3
#include <cstdlib>
#include <cmath>
#else
#include <stdlib.h>
#include <math.h>
#endif
#include "../baseClasses/Data.h"
#include "../templates/Matrix.h"
#include "../templates/DynamicArray.h"
#include "../baseClasses/Random.h"
#include "../baseClasses/Pattern.h"
#include "../baseClasses/PatternList.h"
#include "Variable.h"
#include "Component.h"

class Chromo {
 public:
  Chromo(unsigned classification);
  Chromo(unsigned classification, double coefficient);
  Chromo(Data *, unsigned classification);
  Chromo(const Chromo &);
  ~Chromo();
  void Mutate();
  Chromo *Crossover(Chromo *);
  void Calculate(const DynamicArray<Pattern *>&, double std);
  void CalculateFitness(const DynamicArray<Pattern *>&, double std);
  double Fitness() const;
  double MSE() const;
  double AE() const;
  double SSE() const;
  double CD() const;
  double ACD() const;
  unsigned Complexity() const;
  unsigned Terms() const;
  unsigned GetClassification();
  void MutateShrink();

  bool operator == (const Chromo &) const;
  bool cmp(const Chromo &) const;
  bool operator != (const Chromo &) const;
  Component& ChooseComponent() const;

 private:
  void CalculateComponents(const DynamicArray<Pattern *>&);
  void MutateExpand();
  void MutateComponent();
  void Sort();
  void RemoveBad();

  static Data * data;
  double mse;
  double ae;
  double se;
  double cd;
  double acd;
  unsigned complexity;
  unsigned classification;
  DynamicArray<Component> components;
  friend ostream& operator << (ostream &os, Chromo &chromo);
};

#endif
