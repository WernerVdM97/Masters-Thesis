#include "Variable.h"

Variable::Variable() {
  variable = 0;
  order = 0;
}

Variable::Variable(unsigned a, unsigned char b) {
  variable = a;
  order = b;
}

Variable::Variable(const Variable& v) {
  variable = v.variable;
  order = v.order;
}

bool Variable::operator == (const Variable& v) const {
  if ((v.variable == variable) && (v.order == order))
    //if (v.variable == variable)
    return true;
  return false;
}

bool Variable::operator != (const Variable& v) const {
  if ((v.variable == variable) && (v.order == order))
    //if (v.variable == variable)
    return false;
  return true;
}

bool Variable::operator < (const Variable& v) const {
  if (variable < v.variable)
    return true;
  else if (variable == v.variable)
    if (order < v.order)
      return true;
  return false;
}

bool Variable::operator > (const Variable& v) const {
  if (variable > v.variable)
    return true;
  else if (variable == v.variable)
    if (order > v.order)
      return true;
  return false;
}

Variable::~Variable() {
}

Variable& Variable::operator = (const Variable& v) {
  if (this == &v)
    return *this;
  variable = v.variable;
  order = v.order;
  return *this;
}

ostream& operator << (ostream &os, const Variable& variable) {
  os << "*pow($x" << variable.variable << "," << (unsigned) variable.order <<  ")";
  return os;
}


