#ifndef PATTERN_H
#define PATTERN_H

#include <iostream>
#include <cmath>
#include "../templates/Exceptions.h"

class Pattern {
 public:
  Pattern();
  Pattern(const unsigned, const char *);
  Pattern(const double *, const unsigned, const char *);
  Pattern(const Pattern &);
  ~Pattern();

  Pattern& operator = (const Pattern &);
  Pattern& operator += (const Pattern &);
  Pattern& operator -= (const Pattern &);
  Pattern& SumSqr(const Pattern &);
  Pattern& SubtractSqr(const Pattern &);
  Pattern& CalculateCentroid(const Pattern &, const double&);
  Pattern& CalculateStdDev(const Pattern &, const Pattern &, const double&);
  double Distance(const Pattern &pattern);
  bool CorrectClassification(const Pattern &pattern);
  bool OutBounds(const Pattern &mean, const Pattern &dev);
  double& operator [] (unsigned index);
  bool IsBlank();

 private:
  double *attributes;
  const char *mask;
  unsigned length;

friend ostream& operator << (ostream &, const Pattern &);
friend ostream& operator << (ostream &, const Pattern *);
};

#endif
