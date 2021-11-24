#ifndef DECTREE_H
#define DECTREE_H

#include "../baseClasses/Data.h"
#include "../baseClasses/Random.h"
#include "../templates/DynamicArray.h"
#include "ChromoPool.h"
#include "DecisionNode.h"
#include "NominalAntecedent.h"
#include "ContinuousAntecedent.h"
#include "ContinuousConsequent.h"

class DecisionTree {
 public:
  DecisionTree(Data *, ChromoPool *);
  DecisionTree(const DecisionTree &);
  ~DecisionTree();

  DecisionTree& operator = (const DecisionTree &);
  void Print(ostream&, unsigned, DecisionNode *);
  friend ostream& operator << (ostream&, DecisionTree &);
  unsigned GetNoNodes();

  void Mutate();
  DecisionTree* Crossover(DecisionTree &);
  void Calculate(double, double);
  double GetFitness() const;
  double GetTMSE() const;
  double GetVMSE() const;
  double GetTCD() const;
  double GetVCD() const;
  double GetTACD() const;
  double GetVACD() const;
  double GetTMAE() const;
  double GetVMAE() const;
  void Parse();
  bool ChromoUsed(Chromo *);

  void AddCover(DynamicArray<Pattern *> *);
  void SetCovers(DynamicArray<Pattern *> *, DynamicArray<Pattern *> *);
  void CalculateGeneralization(DynamicArray<Pattern *> *, double);
  double GetGMSE() const;
  double GetGCD() const;
  double GetGACD() const;
  double GetGMAE() const;
  void Optimize();
  void CalculateMiscellaneous(double &, double &, double &);

 private:
  DecisionNode *head;
  static Data *data;
  static ChromoPool *chromoPool;
  double tmse;
  double vmse;
  double gmse;
  double tcd;
  double vcd;
  double gcd;
  double tacd;
  double vacd;
  double gacd;
  double tmae;
  double vmae;
  double gmae;
  unsigned nodes;

  void MutateExpand();
  void MutateShrink();
  void MutateNode();
  DecisionNode* NewDecisionNode(DynamicArray<Pattern *> *, DynamicArray<Pattern *> *, Consequent *);
};

#endif
