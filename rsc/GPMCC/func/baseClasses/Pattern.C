#include "Pattern.h"

/* Class: Pattern
   By:    G. Potgieter

   The pattern class logically represents a pattern from the pattern file.

   essentually the pattern class is an n dimensional double array, that
   has a reference to a type mask.

   For clustering and boundary checking:
     Only-non classification values should be considered

   For classification:
     Only classification values should be considered.

   The centroid should reflect the correct classification and stddev
   therefore they have to be calculated at all times.

   For function optimization, only continuous valued attributes should be 
   considered, nominal valued attributes will be considered by the model
   tree component (strange discontinuities could arise). 
*/

Pattern::Pattern() {
  attributes = NULL;
  length = 0;
  mask = NULL;
}

Pattern::Pattern(const unsigned length, const char *mask) {
  this->length = length;
  attributes = new double[length];
  this->mask = mask;
  for (unsigned i = 0; i < length; i++)
    attributes[i] = 0;
}

Pattern::Pattern(const double *attributes, const unsigned length, const char *mask) {
  this->length = length;
  this->attributes = new double[length];
  this->mask = mask;
  for (unsigned i = 0; i < length; i++)
    this->attributes[i] = attributes[i];
}

Pattern::Pattern(const Pattern &pattern) {
  length = pattern.length;
  attributes = new double[length];
  mask = pattern.mask;
  for (unsigned i = 0; i < length; i++)
    attributes[i] = pattern.attributes[i];
}

Pattern::~Pattern() {
  delete [] attributes;
}

Pattern& Pattern::operator = (const Pattern &pattern) {
  if (&pattern == this)
    return *this;
  if (length != pattern.length) {
    if (attributes != NULL)
      throw new UnmatchedLengthException();
    length = pattern.length;
    attributes = new double[length];
    mask = pattern.mask;
  }
  for (unsigned i = 0; i < length; i++)
    attributes[i] = pattern.attributes[i];
  return *this;
}

Pattern& Pattern::operator += (const Pattern &pattern) {
  if (length != pattern.length) {
    if (attributes != NULL)
      throw new UnmatchedLengthException();
    length = pattern.length;
    attributes = new double[length];
    mask = pattern.mask;
    for (unsigned i = 0; i < length; i++)
      attributes[i] = 0;
  }
  for (unsigned i = 0; i < length; i++)
    attributes[i] += pattern.attributes[i];
  return *this;
}

Pattern& Pattern::operator -= (const Pattern &pattern) {
  if (length != pattern.length) {
    if (attributes != NULL)
      throw new UnmatchedLengthException();
    length = pattern.length;
    attributes = new double[length];
    mask = pattern.mask;
    for (unsigned i = 0; i < length; i++)
      attributes[i] = 0;
  }
  for (unsigned i = 0; i < length; i++)
    attributes[i] -= pattern.attributes[i];
  return *this;
}

Pattern& Pattern::SumSqr(const Pattern &pattern) {
  if (length != pattern.length) {
    if (attributes != NULL)
      throw new UnmatchedLengthException();
    length = pattern.length;
    attributes = new double[length];
    mask = pattern.mask;
    for (unsigned i = 0; i < length; i++)
      attributes[i] = 0;
  }
  for (unsigned i = 0; i < length; i++)
    attributes[i] += pattern.attributes[i] * pattern.attributes[i];
  return *this;
}

Pattern& Pattern::SubtractSqr(const Pattern &pattern) {
  if (length != pattern.length) {
    if (attributes != NULL)
      throw new UnmatchedLengthException();
    length = pattern.length;
    attributes = new double[length];
    mask = pattern.mask;
    for (unsigned i = 0; i < length; i++)
      attributes[i] = 0;
  }
  for (unsigned i = 0; i < length; i++)
    attributes[i] -= pattern.attributes[i] * pattern.attributes[i];
  return *this;
}

Pattern& Pattern::CalculateCentroid(const Pattern &sum, const double &cons) {
  if (length != sum.length) {
    if (attributes != NULL)
      throw new UnmatchedLengthException();
    length = sum.length;
    attributes = new double[length];
    mask = sum.mask;
  }
  if (cons != 0) {
    for (unsigned i = 0; i < length; i++)
      attributes[i] = sum.attributes[i] / cons;
  } else {
    for (unsigned i = 0; i < length; i++)
      attributes[i] = 0.0;
  }
  return *this;
}

Pattern& Pattern::CalculateStdDev(const Pattern &sum, const Pattern &sumSq, const double &cons) {
  if (length != sum.length) {
    if (attributes != NULL)
      throw new UnmatchedLengthException();
    length = sum.length;
    attributes = new double[length];
    mask = sum.mask;
  }
  if (cons > 1) {
    for (unsigned i = 0; i < length; i++)
      attributes[i] = sqrt( (sumSq.attributes[i] - (sum.attributes[i] * sum.attributes[i]) / cons) / (cons - 1.0));
  } else {
    for (unsigned i = 0; i < length; i++)
      attributes[i] = 0.0;
  }
  return *this;
}

double Pattern::Distance(const Pattern &pattern) {
  if (length != pattern.length)
      throw new UnmatchedLengthException();
  double distance = 0;
  double tmp;
  for (unsigned i = 0; i < length; i++) {
    if ((mask[i] & 2) == 2)
      continue;
    tmp = attributes[i] - pattern.attributes[i];
    distance += tmp * tmp;
  }
  return distance;
  //return sqrt(distance);
}

bool Pattern::CorrectClassification(const Pattern &pattern) {
  for (unsigned i = 0; i < length; i++) {
    if ((mask[i] & 2) == 0)
      continue;
    if (pattern.attributes[i] - attributes[i] > 0.5)
      return false;
    if (pattern.attributes[i] - attributes[i] < -0.5)
      return false;
  }
  return true;
}

bool Pattern::OutBounds(const Pattern &mean, const Pattern &dev) {
  if ((length != mean.length) || (length != dev.length))
      throw new UnmatchedLengthException();
  bool test = false;
  for (unsigned i = 0; i < length; i++) {
    if ((mask[i] & 2) == 2)
      continue;    
    test = ((attributes[i] < mean.attributes[i] - dev.attributes[i]) 
	    || (attributes[i] > mean.attributes[i] + dev.attributes[i]));
    if (test)
      return test;
  }
  return false;
}

double& Pattern::operator [] (unsigned index) {
  return attributes[index];
}

bool Pattern::IsBlank() {
  return (attributes == NULL);
}

ostream& operator << (ostream &os, const Pattern &pattern) {
  os << &pattern;
  return os;
}

ostream& operator << (ostream &os, const Pattern *pattern) {
  if (pattern == NULL)
    os << "null";
  else {
    for (unsigned i = 0; i < pattern->length; i++) {
      os << pattern->attributes[i];
      if (i != pattern->length - 1)
	os << " ";
    }
  }
  os.flush();
  return os;
}
