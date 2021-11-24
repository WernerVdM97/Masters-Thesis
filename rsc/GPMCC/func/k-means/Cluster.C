/* Class: Cluster
   By:    G. Potgieter

   Part of a k-means clustering implementation.

   Essentially a container for patterns, implementing an extremely efficient 
   direct update method (as apposed to batch), and pattern distance metric.
*/

#include "Cluster.h"

Cluster::Cluster() {
}

Cluster::~Cluster() {
}

void Cluster::AddPattern(Pattern *pattern) {
  patternList.AddPattern(pattern);
  centroidSum += *pattern;
  centroidSumSq.SumSqr(*pattern);
  centroid.CalculateCentroid(centroidSum, patternList.Length());
  stdDev.CalculateStdDev(centroidSum, centroidSumSq, patternList.Length());
}

void Cluster::RemovePattern(Pattern *pattern) {
  patternList.RemovePattern();
  centroidSum -= *pattern;
  centroidSumSq.SubtractSqr(*pattern);
  centroid.CalculateCentroid(centroidSum, patternList.Length());
  stdDev.CalculateStdDev(centroidSum, centroidSumSq, patternList.Length());
}

void Cluster::AddValidationPattern(Pattern *pattern) {
  validationPatterns.AddPattern(pattern);
}

void Cluster::RemoveValidationPattern(Pattern *pattern) {
  validationPatterns.RemovePattern();
}

void Cluster::TestCloser(Cluster *clusters, int size) {
  if (patternList.Length() > 0) {
    patternList.ToFront();
    do {
      Pattern* tmp = patternList.GetPattern();
      double bdistance = tmp->Distance(centroid);
      int bcluster = -1;
      if (tmp->OutBounds(centroid, stdDev)) {
	double distance;
	for (int j = 1; j < size; j++) {
	  if (this != &clusters[j]) {
	    distance = tmp->Distance(clusters[j].centroid);
	    if (bdistance > distance) {
	      bdistance = distance;
	      bcluster = j;
	    }
	  }
	}
	if (bcluster != -1) {
	  RemovePattern(tmp);
	  clusters[bcluster].AddPattern(tmp);
	}
      }
    } while (patternList.Next());
  }
}

void Cluster::TestValidationCloser(Cluster *clusters, int size) {
  if (validationPatterns.Length() > 0) {
    validationPatterns.ToFront();
    do {
      Pattern* tmp = validationPatterns.GetPattern();
      double bdistance = tmp->Distance(centroid);
      int bcluster = -1;
      if (tmp->OutBounds(centroid, stdDev)) {
	double distance;
	for (int j = 1; j < size; j++) {
	  if (this != &clusters[j]) {
	    distance = tmp->Distance(clusters[j].centroid);
	    if (bdistance > distance) {
	      bdistance = distance;
	      bcluster = j;
	    }
	  }
	}
	if (bcluster != -1) {
	  RemoveValidationPattern(tmp);
	  clusters[bcluster].AddValidationPattern(tmp);
	}
      }
    } while (validationPatterns.Next());
  }
}

void Cluster::Finalize() {
  patternList.Finalize();
  validationPatterns.Finalize();
}

Pattern* Cluster::operator [] (unsigned index) const {
  return patternList[index];
}

Pattern* Cluster::GetVPattern(unsigned index) const {
  return validationPatterns[index];
}

unsigned Cluster::Length() const {
  return patternList.Length();
}

unsigned Cluster::ValidationLength() const {
  return validationPatterns.Length();
}

Pattern Cluster::GetCentroid() const {
  return centroid;
}

Pattern Cluster::GetStdDev() const {
  return stdDev;
}

double Cluster::GetQE() {
  double distance = 0;
  if (patternList.Length() > 0) {
    patternList.ToFront();
    do {
      distance += sqrt(patternList.GetPattern()->Distance(centroid));
    } while (patternList.Next());
    //distance /= patternList.Length();
  }
  return distance;
}

double Cluster::GetVQE() {
  double distance = 0;
  if (validationPatterns.Length() > 0) {
    validationPatterns.ToFront();
    do {
      distance += sqrt(validationPatterns.GetPattern()->Distance(centroid));
    } while (validationPatterns.Next());
    //distance /= patternList.Length();
  }
  return distance;
}

double Cluster::GetClassification() {
  double clas = 0;
  if (patternList.Length() > 0) {
    patternList.ToFront();
    do {
      if (patternList.GetPattern()->CorrectClassification(centroid))
	clas += 1.0;
    } while (patternList.Next());
    clas /= patternList.Length();
  }
  return clas;
}

double Cluster::GetVClassification() {
  double clas = 0;
  if (validationPatterns.Length() > 0) {
    validationPatterns.ToFront();
    do {
      if (validationPatterns.GetPattern()->CorrectClassification(centroid))
	clas += 1.0;
    } while (validationPatterns.Next());
    clas /= validationPatterns.Length();
  }
  return clas;
}

PatternList& Cluster::GetPatternList() {
  return this->patternList;
}

ostream& operator << (ostream &os, const Cluster &c) {
  os << c.centroid;
  return os;
}


