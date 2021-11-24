#ifndef DECNOD_H
#define DECNOD_H

#include "../templates/Exceptions.h"
#include "../gapolycurve/Chromo.h"
#include "Antecedent.h"
#include "Consequent.h"

class DecisionNode {
 public:
  DecisionNode(Consequent *);
  DecisionNode(Antecedent *, DecisionNode *, DecisionNode *);
  DecisionNode(const DecisionNode &);
  ~DecisionNode();
  bool IsConsequent();
  bool SetConsequent(Consequent *);
  bool SetAntecedent(Antecedent *);
  bool SetAffirmative(DecisionNode *);
  bool SetNegative(DecisionNode *);
  Antecedent* GetAntecedent() const;
  Consequent* GetConsequent() const;
  DecisionNode* GetAffirmative() const;
  DecisionNode* GetNegative() const;

  friend ostream& operator << (ostream&, DecisionNode &);

  unsigned GetNoNodes() const;
  void ObtainLeafAntecedents(DynamicArray<DecisionNode *> &) const;
  void ObtainNonLeafAntecedents(DynamicArray<DecisionNode *> &) const;
  void ObtainAntecedentNodes(DynamicArray<DecisionNode *> &) const;
  void ObtainConsequentNodes(DynamicArray<DecisionNode *> &) const;
  void ObtainAllNodes(DynamicArray<DecisionNode *> &) const;
  DecisionNode* ParseNode();
  void SetCovers(DynamicArray<Pattern *> *, DynamicArray<Pattern *> *);
  void AddCover(DynamicArray<Pattern *> *);
  void GetCovers(DynamicArray<Pattern *> *, DynamicArray<Pattern *> *);
  void CalculateFitness(double &, double &, unsigned &, double &, double &);
  void CalculateMiscellaneous(unsigned &, unsigned &, unsigned &, unsigned);

  double ObtainWorstLeafAntecedent(DynamicArray<DecisionNode *> &) const;
  void ObtainWorstConsequent(DynamicArray<DecisionNode *> &) const;
  bool ChromoUsed(Chromo *);
  void CalculateGeneralization(DynamicArray<Pattern *> *, double &, unsigned &, double &);
  void Optimize();

 private:
  Antecedent *antecedent;
  Consequent *consequent;
  DecisionNode *affirmative;
  DecisionNode *negative;
};

#endif
