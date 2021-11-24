#ifndef VAR_H
#define VAR_H

#include <iostream>

using namespace std;

class Variable {
 public:
  unsigned variable;
  unsigned char order;

  Variable();
  Variable(unsigned, unsigned char);
  Variable(const Variable&);
  ~Variable();
  bool operator == (const Variable&) const;
  bool operator != (const Variable&) const;
  bool operator < (const Variable&) const;
  bool operator > (const Variable&) const;
  Variable& operator = (const Variable&);
  double GetEvaluation(unsigned) const;
  
  friend ostream& operator << (ostream &os, const Variable& variable);
};

#endif

