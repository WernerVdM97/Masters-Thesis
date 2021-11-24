#ifndef ANTE_H
#define ANTE_H

#include "../baseClasses/Data.h"
#include "../baseClasses/Pattern.h"
#include "../templates/Exceptions.h"
#include "../templates/DynamicArray.h"
#include <iostream>

class Antecedent {
 public:
  Antecedent(Data *);
  Antecedent(const Antecedent &);
  virtual ~Antecedent();

  virtual void Mutate() = 0;
  virtual ostream& Print(ostream&) = 0;
  friend ostream& operator << (ostream &, Antecedent &);
  virtual Antecedent* Copy() = 0;
  virtual bool CoverPattern(Pattern&) const = 0;

 protected:
  static Data *data;

};

#endif
