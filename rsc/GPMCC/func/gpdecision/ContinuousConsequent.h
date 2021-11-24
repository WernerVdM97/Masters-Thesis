#ifndef CONCONS_H
#define CONCONS_H

#include <iostream>
#include "../baseClasses/Data.h"
#include "../k-means/Clusterer.h"
#include "../gapolycurve/Chromo.h"
#include "../gapolycurve/GAPolyCurve.h"
#include "Consequent.h"
#include "ChromoPool.h"

class ContinuousConsequent : public Consequent {
 public:
  ContinuousConsequent(Data *, ChromoPool *, DynamicArray<Pattern *> *, DynamicArray<Pattern *> *);
  ContinuousConsequent(const ContinuousConsequent &);
  virtual ~ContinuousConsequent();
  
  virtual void Mutate();
  virtual ostream& Print(ostream&);
  friend ostream& operator << (ostream &, ContinuousConsequent &);
  virtual Consequent* Copy();
  virtual void Optimize();
  virtual void CalculateFitness();
  virtual bool InUse(Chromo *chromo);
  virtual void CalculateGeneralization(DynamicArray<Pattern *> *, double &, double &);

 protected:
  Chromo *chromo;

 private:
  static ChromoPool *chromoPool;

  double CalculateStd(DynamicArray<Pattern *> *, unsigned);
};

#endif
