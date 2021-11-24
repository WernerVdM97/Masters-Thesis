#ifndef CLUSTER_H
#define CLUSTER_H

#include "../baseClasses/Pattern.h"
#include "../baseClasses/PatternList.h"

class Cluster {
 public:
  Cluster();
  ~Cluster();

  void AddPattern(Pattern *);
  void AddValidationPattern(Pattern *);
  void RemovePattern(Pattern *);
  void RemoveValidationPattern(Pattern *);
  void TestCloser(Cluster *cluster, int);
  void TestValidationCloser(Cluster *cluster, int);
  void Finalize();
  unsigned Length() const;
  unsigned ValidationLength() const;
  Pattern GetCentroid() const;
  Pattern GetStdDev() const;
  Pattern* operator [] (unsigned index) const;
  Pattern* GetVPattern(unsigned index) const;
  double GetQE();
  double GetVQE();
  double GetClassification();
  double GetVClassification();
  PatternList& GetPatternList();

 private:
  PatternList patternList;
  PatternList validationPatterns;
  Pattern centroid;
  Pattern centroidSum;
  Pattern centroidSumSq;
  Pattern stdDev;

  friend ostream& operator << (ostream &, const Cluster &);
};

#endif
