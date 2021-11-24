#ifndef COMPONENT_H
#define COMPONENT_H

#include <cstdlib>
#include <cmath>
#include <iostream>
#include "../baseClasses/Data.h"
#include "../templates/DynamicArray.h"
#include "Variable.h"
#include "../baseClasses/Random.h"
#include "../baseClasses/Pattern.h"

class Component {
 public:
  Component();
  Component(unsigned, unsigned char);
  Component(Data *);
  Component(const Component &);
  ~Component();

  bool operator == (const Component&) const;
  bool operator != (const Component&) const;
  bool operator < (const Component&) const;
  bool operator > (const Component&) const;
  Component& operator = (const Component&);

  void SetCoefficient(double);
  double GetCoefficient() const;
  double MatrixEvaluate(Pattern &) const;
  double Evaluate(Pattern &) const;
  unsigned Complexity() const;
  void Mutate();
  bool isEmpty();

 private:
  static Data *data;
  DynamicArray<Variable> variables;
  double coefficient;

  double T(const double &, const unsigned &) const;
  void Rationalize();

 friend ostream& operator << (ostream &os, const Component& component);
};

#endif



