/* Class: Component
   By:    G. Potgieter

   Each component represents a list of variables and their orders as well
   as one coefficient.

   The representation might look something like:

   C*T1(x1)*T2(x5)

   Which, corresponds to:

   C*x1*x5^2

   if we are using normal polynomials or

   C*x1*(x5^2 - 1/2)

   if we are using Normalized Chebyshev Polynomials.
*/

#include "Component.h"

Data *Component::data = NULL;

Component::Component() {
  coefficient = 0;
}

Component::Component(Data *data) {
  Random random;
  Component::data = data;

  unsigned polynomialOrder = data->GetPolynomialOrder();
  unsigned noAttributes = data->GetNoAttributes();
  char *mask = data->GetAttributeMask();

  unsigned char po = (unsigned char) random.GetRandom(0U, polynomialOrder + 1U);

  unsigned att;
  unsigned char pot;
  for (unsigned k = 0; ((k < 50) && (po > 0)); k++) {
    unsigned l = 0;
    // Can only select continuous attributes
    do {
      att = random.GetRandom(0U, noAttributes);
      l++;
    } while ((mask[att] != 1) && (l < 50));
    // breaks if got stuck
    if (l == 50)
      break;

    pot = (unsigned char) random.GetRandom(1U, po + 1U);
    Variable tmp(att,pot);
    if (variables.InsertUniqueSorted(tmp))
      po -= pot;
  }
  coefficient = 0;
  Rationalize();
}

Component::Component(unsigned i, unsigned char j) {
  Variable tmp(i,j);
  variables.Add(tmp);
  coefficient = 0;
}

Component::Component(const Component& component) {
  coefficient = component.coefficient;
  variables = component.variables;
  data = component.data;
}

Component::~Component() {
  //cout << "<component> deleted\n";
}

bool Component::operator == (const Component& component) const {
  if (variables.Length() != component.variables.Length())
    return false;
  for (unsigned i = 0; i < variables.Length(); i++)
    if (variables[i] != component.variables[i])
      return false;
  return true;
}

bool Component::operator != (const Component& component) const {
  if (variables.Length() != component.variables.Length())
    return true;
  for (unsigned i = 0; i < variables.Length(); i++)
    if (variables[i] != component.variables[i])
      return true;
  return false;
}

bool Component::operator < (const Component& component) const {
  for (unsigned i = 0; ((i < variables.Length()) && (i < component.variables.Length())); i++)
    if (variables[i] > component.variables[i])
      return false;
    else if (variables[i] < component.variables[i])
      return true;
  if (component.variables.Length() > variables.Length())
    return true;
  return false;
}

bool Component::operator > (const Component& component) const{
  for (unsigned i = 0; ((i < variables.Length()) && (i < component.variables.Length())); i++)
    if (variables[i] < component.variables[i])
      return false;
    else if (variables[i] > component.variables[i])
      return true;
  if (component.variables.Length() < variables.Length())
    return true;
  return false;
}

Component& Component::operator = (const Component& component) {
  coefficient = component.coefficient;
  variables = component.variables;
  return *this;
}

void Component::SetCoefficient(double c) {
  coefficient = c;
}

double Component::GetCoefficient() const {
  return coefficient;
}

ostream& operator << (ostream &os, const Component& component) {
  os << component.coefficient;
  for (unsigned i = 0; i < component.variables.Length(); i++) {
    os << "*pow(";
    if (Component::data->GetSyntaxMode() == 0)
      os << "$";
    os << Component::data->GetAttribute(component.variables[i].variable) << "," << (unsigned) component.variables[i].order << ")";
  }
  return os;
}

double Component::T(const double &base, const unsigned &exponent) const {
  // Normal copmonents
  
  double val = 1;
  for (unsigned i = 0; i < exponent; i++)
    val *= base;
  return val;
  

  // Normalized Chebyshev components
  /*
  switch (exponent) {
  default: return 1;
  case 1: return base;
  case 2: return base * base - 0.5;
  case 3: return base * base * base - 0.75 * base;
  case 4: return base * base * base * base - base * base + 0.125;
  case 5: return base * base * base * base * base - 1.25 * base * base * base + 0.5 * base;
  }
  */
}

double Component::MatrixEvaluate(Pattern &pattern) const {
  double t = 1;
  unsigned length = variables.Length();
  for (unsigned j = 0; j < length; j++)
    t *= T(pattern[variables[j].variable], variables[j].order);
  return t;
}

double Component::Evaluate(Pattern &pattern) const {
  double t = coefficient;
  unsigned length = variables.Length();
  for (unsigned j = 0; j < length; j++)
    t *= T(pattern[variables[j].variable], variables[j].order);
  return t;
}

unsigned Component::Complexity() const {
  unsigned length = variables.Length();
  if (length == 0)
    return 1;
  unsigned complexity = 0;
  for (unsigned j = 0; j < length; j++)
    complexity += variables[j].order;
  return complexity;
}

void Component::Mutate() {
  if (variables.Length() > 0) {
    Random random;
    unsigned item = random.GetRandom(0, variables.Length());
    Variable var = variables[item];
    variables.Delete(item);

    unsigned length = variables.Length();
    unsigned orders = 0;
    for (unsigned i = 0; i < length; i++) {
      orders += variables[i].order;
    }
    unsigned char po = (unsigned char) random.GetRandom(0U, Component::data->GetPolynomialOrder() - orders + 1U);
    if (po > 0) {
      var.order = po;
      unsigned noAttributes = Component::data->GetNoAttributes();
      char *mask = Component::data->GetAttributeMask();
      unsigned l = 0;
      unsigned att;
      do {
	att = random.GetRandom(0U, noAttributes);
	l++;
      } while ((mask[att] != 1) && (l < 50));
      if (l != 50)
	var.variable = att;
      variables.InsertUniqueSorted(var);
    }
  }
  Rationalize();
}

void Component::Rationalize() {
  for (unsigned i = 1; i < variables.Length(); i++) {
    if (variables[i].variable == variables[i - 1].variable) {
      variables[i - 1].order += variables[i].order;
      variables.Delete(i);
      i--;
    }
  }
}

bool Component::isEmpty() {
  if (variables.Length() == 0)
    return true;
  return false;
}
