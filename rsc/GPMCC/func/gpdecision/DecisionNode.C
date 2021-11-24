#include "DecisionNode.h"

DecisionNode::DecisionNode(Consequent *consequent) {
  if (consequent == NULL)
    throw new NullPointerException();
  this->consequent = consequent;
  this->antecedent = NULL;
  this->affirmative = NULL;
  this->negative = NULL;
}

DecisionNode::DecisionNode(Antecedent *antecedent, DecisionNode *affirmative, DecisionNode *negative) {
  if ((antecedent == NULL) || (affirmative == NULL) || (negative == NULL))
    throw new NullPointerException();
  this->antecedent = antecedent;
  this->consequent = NULL;
  this->affirmative = affirmative;
  this->negative = negative;
}

DecisionNode::DecisionNode(const DecisionNode &dn) {
  if (this == &dn)
    return;
  if (dn.consequent != NULL) {
    consequent = dn.consequent->Copy();
    antecedent = NULL;
    affirmative = NULL;
    negative = NULL;
  } else {
    consequent = NULL;
    antecedent = dn.antecedent->Copy();
    affirmative = new DecisionNode(*(dn.affirmative));
    negative = new DecisionNode(*(dn.negative));
  }
}

DecisionNode::~DecisionNode() {
  if (consequent != NULL) {
    delete consequent;
  } else {
    delete antecedent;
    delete affirmative;
    delete negative;
  }
  antecedent = NULL;
  consequent = NULL;
  affirmative = NULL;
  negative = NULL;
}

unsigned DecisionNode::GetNoNodes() const {
  if (consequent != NULL)
    return 1;
  return 1 + affirmative->GetNoNodes() + negative->GetNoNodes();
}

bool DecisionNode::IsConsequent() {
  return (consequent != NULL);
}

bool DecisionNode::SetAntecedent(Antecedent *a) {
  if (a == NULL)
    throw new Exception("antecedent is NULL in DecisionNode, SetAntecedent");
  if (antecedent != NULL) {
    delete antecedent;
    antecedent = a;
    return true;
  }
  return false;
}

bool DecisionNode::SetConsequent(Consequent *c) {
  if (c == NULL)
    throw new Exception("consequent is NULL in DecisionNode, SetConsequent");
  if (consequent != NULL) {
    delete consequent;
    consequent = c;
    return true;
  }
  return false;
}

bool DecisionNode::SetAffirmative(DecisionNode *dn) {
  if (dn == NULL)
    throw new Exception("decisionNode is NULL in DecisionNode, SetAffirmative");
  if (consequent == NULL) {
    if (affirmative != NULL)
      delete affirmative;
    affirmative = dn;
    return true;
  }
  return false;
}

bool DecisionNode::SetNegative(DecisionNode *dn) {
  if (dn == NULL)
    throw new Exception("decisionNode is NULL in DecisionNode, SetNegative");
  if (consequent == NULL) {
    if (negative != NULL)
      delete negative;
    negative = dn;
    return true;
  }
  return false;
}

Antecedent* DecisionNode::GetAntecedent() const {
  return antecedent;
}

Consequent* DecisionNode::GetConsequent() const {
  return consequent;
}

DecisionNode* DecisionNode::GetAffirmative() const {
  return affirmative;
}

DecisionNode* DecisionNode::GetNegative() const {
  return negative;
}

void DecisionNode::ObtainLeafAntecedents(DynamicArray<DecisionNode *> &nodes) const {
  if (consequent != NULL)
    return;

  if ((affirmative->consequent != NULL) || (negative->consequent != NULL))
    nodes.Add((DecisionNode *) this);
  affirmative->ObtainLeafAntecedents(nodes);
  negative->ObtainLeafAntecedents(nodes);
}

void DecisionNode::ObtainNonLeafAntecedents(DynamicArray<DecisionNode *> &nodes) const {
  if (consequent != NULL)
    return;

  if ((affirmative->consequent == NULL) || (negative->consequent == NULL))
    nodes.Add((DecisionNode *) this);
  affirmative->ObtainNonLeafAntecedents(nodes);
  negative->ObtainNonLeafAntecedents(nodes);
}

void DecisionNode::ObtainAntecedentNodes(DynamicArray<DecisionNode *> &nodes) const {
  if (consequent != NULL)
    return;
  nodes.Add((DecisionNode *) this);
  affirmative->ObtainAntecedentNodes(nodes);
  negative->ObtainAntecedentNodes(nodes);
}

void DecisionNode::ObtainConsequentNodes(DynamicArray<DecisionNode *> &nodes) const {
  if (consequent != NULL) {
    nodes.Add((DecisionNode *) this);
    return;
  }
  affirmative->ObtainConsequentNodes(nodes);
  negative->ObtainConsequentNodes(nodes);
}

void DecisionNode::ObtainAllNodes(DynamicArray<DecisionNode *> &nodes) const {
  nodes.Add((DecisionNode *) this);
  if (consequent != NULL)
    return;
  affirmative->ObtainAllNodes(nodes);
  negative->ObtainAllNodes(nodes);
}

DecisionNode* DecisionNode::ParseNode() {
  if (consequent != NULL) {
    return NULL;
  }

  DecisionNode *aff = affirmative->ParseNode();
  DecisionNode *neg = negative->ParseNode();

  if (aff != NULL) {
    DecisionNode *tmp = new DecisionNode(*aff);
    delete affirmative;
    affirmative = tmp;
  }
  if (neg != NULL) {
    DecisionNode *tmp = new DecisionNode(*neg);
    delete negative;
    negative = tmp;
  }

  if ((affirmative->consequent != NULL) 
      && (affirmative->consequent->GetNoTrainingCover() == 0))
    return negative;
  else if ((negative->consequent != NULL) 
	   && (negative->consequent->GetNoTrainingCover() == 0))
    return affirmative;
  return NULL;
}

void DecisionNode::SetCovers(DynamicArray<Pattern *> *tCover, DynamicArray<Pattern *> *vCover) {
  if (consequent != NULL) {
    consequent->SetCovers(tCover, vCover);
   } else {
    DynamicArray<Pattern *> *tCoverA = new DynamicArray<Pattern *>();
    DynamicArray<Pattern *> *tCoverN = new DynamicArray<Pattern *>();
    DynamicArray<Pattern *> *vCoverA = new DynamicArray<Pattern *>();
    DynamicArray<Pattern *> *vCoverN = new DynamicArray<Pattern *>();
    
    unsigned length = tCover->Length();
    for (unsigned i = 0; i < length; i++) {
      Pattern *pattern = (*tCover)[i];
      if (antecedent->CoverPattern(*pattern))
	tCoverA->Add(pattern);
      else
	tCoverN->Add(pattern);
    }
    delete tCover;
    length = vCover->Length();
    for (unsigned i = 0; i < length; i++) {
      Pattern *pattern = (*vCover)[i];
      if (antecedent->CoverPattern(*pattern))
	vCoverA->Add(pattern);
      else
	vCoverN->Add(pattern);
    }
    delete vCover;
    affirmative->SetCovers(tCoverA, vCoverA);
    negative->SetCovers(tCoverN, vCoverN);
  }
}

void DecisionNode::AddCover(DynamicArray<Pattern *> *tCover) {
  if (consequent != NULL) {
    consequent->AddCover(tCover);
    delete tCover;
   } else {
    DynamicArray<Pattern *> *tCoverA = new DynamicArray<Pattern *>();
    DynamicArray<Pattern *> *tCoverN = new DynamicArray<Pattern *>();
    unsigned length = tCover->Length();
    for (unsigned i = 0; i < length; i++) {
      Pattern *pattern = (*tCover)[i];
      if (antecedent->CoverPattern(*pattern))
	tCoverA->Add(pattern);
      else
	tCoverN->Add(pattern);
    }
    delete tCover;
    if (tCoverA->Length() > 0)
      affirmative->AddCover(tCoverA);
    else
      delete tCoverA;
    if (tCoverN->Length() > 0)
      negative->AddCover(tCoverN);
    else
      delete tCoverN;
  }
}

void DecisionNode::GetCovers(DynamicArray<Pattern *> *tCover, DynamicArray<Pattern *> *vCover) {
  if (consequent != NULL) {
    consequent->GetCovers(tCover, vCover);
    return;
  }
  affirmative->GetCovers(tCover, vCover);
  negative->GetCovers(tCover, vCover);
}

void DecisionNode::CalculateFitness(double &tsse, double &vsse, 
				    unsigned &complexity, 
				    double &tae, double &vae) {
  if (consequent != NULL) {
    consequent->CalculateFitness();
    tsse += consequent->GetTSSE();
    vsse += consequent->GetVSSE();
    tae += consequent->GetTAE();
    vae += consequent->GetVAE();
    complexity += consequent->GetComplexity();
    return;
  }
  affirmative->CalculateFitness(tsse, vsse, complexity, tae, vae);
  negative->CalculateFitness(tsse, vsse, complexity, tae, vae);
}

void DecisionNode::CalculateMiscellaneous(unsigned &rules, unsigned &conditions, unsigned &terms, unsigned depth) {
  if (consequent != NULL) {
    rules++;
    conditions += depth;
    terms += consequent->GetTerms();
    return;
  }
  affirmative->CalculateMiscellaneous(rules, conditions, terms, depth + 1);
  negative->CalculateMiscellaneous(rules, conditions, terms, depth + 1);
}


double DecisionNode::ObtainWorstLeafAntecedent(DynamicArray<DecisionNode *> &nodes) const {
  if (consequent != NULL) {
    consequent->CalculateFitness();
    return consequent->GetTMSE();
  }
  double lfit = affirmative->ObtainWorstLeafAntecedent(nodes);
  double rfit = negative->ObtainWorstLeafAntecedent(nodes);

  if ((affirmative->consequent != NULL) && (negative->consequent != NULL)) {
      nodes.Add((DecisionNode *) this);
  } else if ((affirmative->consequent != NULL) && (lfit > rfit)) {
      nodes.Add((DecisionNode *) this);
      return lfit;
  } else if ((negative->consequent != NULL) && (rfit >= lfit)) {
      nodes.Add((DecisionNode *) this);
      return rfit;
  }
  if (rfit >= lfit)
    return rfit;
  else
    return lfit;
}

void DecisionNode::ObtainWorstConsequent(DynamicArray<DecisionNode *> &nodes) const {
  if (consequent != NULL) {
    consequent->CalculateFitness();
    if (nodes.Length() == 0) {
      nodes.Add((DecisionNode *) this);
    } else if (consequent->GetTMSE() >= nodes[nodes.Length() - 1]->consequent->GetTMSE()) {
      nodes.Add((DecisionNode *) this);
    }
    return;
  }
  affirmative->ObtainWorstConsequent(nodes);
  negative->ObtainWorstConsequent(nodes);
}

bool DecisionNode::ChromoUsed(Chromo *chromo) {
  if (consequent != NULL)
    return consequent->InUse(chromo);
  bool a = affirmative->ChromoUsed(chromo);
  if (a)
    return a;
  bool n = negative->ChromoUsed(chromo);
  return n;
}

void DecisionNode::CalculateGeneralization(DynamicArray<Pattern *> *gCover, double &sse, unsigned &complexity, double &ae) {
  if (consequent != NULL) {
    consequent->CalculateGeneralization(gCover, sse, ae);
    complexity += consequent->GetComplexity();
    delete gCover;
    return;
  }
  DynamicArray<Pattern *> *gCoverA = new DynamicArray<Pattern *>();
  DynamicArray<Pattern *> *gCoverN = new DynamicArray<Pattern *>();
  unsigned length = gCover->Length();
  for (unsigned i = 0; i < length; i++) {
    Pattern *pattern = (*gCover)[i];
    if (antecedent->CoverPattern(*pattern))
      gCoverA->Add(pattern);
    else
      gCoverN->Add(pattern);
  }
  delete gCover;
  if (gCoverA->Length() > 0)
    affirmative->CalculateGeneralization(gCoverA, sse, complexity, ae);
  else
    delete gCoverA;
  if (gCoverN->Length() > 0)
    negative->CalculateGeneralization(gCoverN, sse, complexity, ae);
  else
    delete gCoverN;
}

void DecisionNode::Optimize() {
  if (consequent != NULL) {
    consequent->Optimize();
    return;
  }
  affirmative->Optimize();
  negative->Optimize();
}
