#ifndef CONANTE_H
#define CONANTE_H

#include "../baseClasses/Data.h"
#include "../baseClasses/Pattern.h"
#include <iostream>
#include "Antecedent.h"

class ContinuousAntecedent : public Antecedent {
 public:
  ContinuousAntecedent(Data *);
  ContinuousAntecedent(const ContinuousAntecedent &);
  virtual ~ContinuousAntecedent();

  virtual void Mutate();
  virtual ostream& Print(ostream&);
  friend ostream& operator << (ostream &, ContinuousAntecedent &);
  virtual Antecedent* Copy();
  virtual bool CoverPattern(Pattern&) const;

 private:
  unsigned attribute;
  char cmp;
  double value;
};

#endif
