#ifndef CONS_H
#define CONS_H

#include <iostream>
#include "../templates/DynamicArray.h"
#include "../baseClasses/Data.h"
#include "../gapolycurve/Chromo.h"

class Consequent {
 public:
  Consequent(Data *data, DynamicArray<Pattern *> *, DynamicArray<Pattern *> *);
  Consequent(const Consequent &);
  virtual ~Consequent();

  virtual void Mutate() = 0;
  virtual ostream& Print(ostream&) = 0;
  friend ostream& operator << (ostream &, Consequent &);
  virtual Consequent* Copy() = 0;
  unsigned GetNoTrainingCover();
  unsigned GetNoValidationCover();
  void SetCovers(DynamicArray<Pattern *> *, DynamicArray<Pattern *> *);
  void AddCover(DynamicArray<Pattern *> *);
  void GetCovers(DynamicArray<Pattern *> *, DynamicArray<Pattern *> *);
  virtual void CalculateFitness() = 0;
  virtual void Optimize() = 0;
  double GetTSSE() const;
  double GetTMSE() const;
  double GetTMAE() const;
  double GetTAE() const;
  double GetVSSE() const;
  double GetVMSE() const;
  double GetVMAE() const;
  double GetVAE() const;

  unsigned GetComplexity() const;
  unsigned GetTerms() const;

  virtual bool InUse(Chromo *) = 0;
  virtual void CalculateGeneralization(DynamicArray<Pattern *> *, double &, double &) = 0;

 protected:
  static Data *data;
  double tsse;
  double tmse;
  double tae;
  double vsse;
  double vmse;
  double vae;

  unsigned complexity;
  unsigned terms;
  DynamicArray<Pattern *> *trainingCover;
  DynamicArray<Pattern *> *validationCover;
};

#endif
