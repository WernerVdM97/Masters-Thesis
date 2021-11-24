#ifndef RANDOM_H
#define RANDOM_H

#include <cstdlib>
#include <ctime>
#include <cmath>

class Random {
 public:
  static void Seed();
  static unsigned GetRandom(unsigned lower, unsigned upper);
  static double GetRandom(double lower, double upper);
  static double GetGaussRandom(double std);
};

#endif
