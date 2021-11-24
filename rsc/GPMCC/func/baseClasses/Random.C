/* Class: Random
   By:    G. Potgieter

   Finally a way to get rid of that pesky range controlling of random numbers.
*/

#include "Random.h"

void Random::Seed() {
  srand(time(NULL));
}

unsigned Random::GetRandom(unsigned lower, unsigned upper) {
  return ((unsigned) (((double) (upper - lower)) * rand() / (RAND_MAX + 1.0) + (double) lower));
}

double Random::GetRandom(double lower, double upper) {
  return ((upper - lower) * rand() / (RAND_MAX + 1.0) + lower);
}

double Random::GetGaussRandom(double std) {
  double x = (double) rand() / (RAND_MAX + 1.0);
  return exp(-1*x*x / (2*std*std));
}
