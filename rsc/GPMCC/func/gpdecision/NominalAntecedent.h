#ifndef NOMANTE_H
#define NOMANTE_H

#include "../baseClasses/Data.h"
#include "../baseClasses/Random.h"
#include "../baseClasses/Pattern.h"
#include <iostream>
#include "Antecedent.h"

class NominalAntecedent : public Antecedent {
 public:
  NominalAntecedent(Data *);
  NominalAntecedent(const NominalAntecedent &);
  virtual ~NominalAntecedent();

  virtual void Mutate();
  virtual ostream& Print(ostream&);
  friend ostream& operator << (ostream &, NominalAntecedent &);
  virtual Antecedent* Copy();
  virtual bool CoverPattern(Pattern&) const;

 private:
  unsigned attribute;
};

#endif
