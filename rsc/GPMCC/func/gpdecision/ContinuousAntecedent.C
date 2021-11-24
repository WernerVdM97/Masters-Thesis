#include "ContinuousAntecedent.h"

ContinuousAntecedent::ContinuousAntecedent(Data *data) : Antecedent(data) {
  char *mask = Antecedent::data->GetAttributeMask();
  unsigned attributes = Antecedent::data->GetNoAttributes();
  do {
    attribute = Random::GetRandom(0U, attributes);
  } while (mask[attribute] != 1);
  if (Random::GetRandom(0.0, 1.0) < 0.1)
    cmp = (char) Random::GetRandom(0U, 2U) * 3;
  else {
    cmp = (char) Random::GetRandom(0U, 2U) + 1;
  }
  cmp = (char) Random::GetRandom(0U, 4U);
  value = Random::GetRandom(Antecedent::data->GetMinimum(attribute), Antecedent::data->GetMaximum(attribute));
}

ContinuousAntecedent::ContinuousAntecedent(const ContinuousAntecedent &ca) : Antecedent(ca) {
  attribute = ca.attribute;
  cmp = ca.cmp;
  value = ca.value;
}

ContinuousAntecedent::~ContinuousAntecedent() {
}

void ContinuousAntecedent::Mutate() {
  double rand = Random::GetRandom(0.0, 1.0);
  if (rand > data->GetDecisionCAConditionOptimize() + data->GetDecisionCAAttributeOptimize()) {
    if (Random::GetRandom(0.0, 1.0) < data->GetDecisionCAClassVsGaussian())
      value += Random::GetRandom(Antecedent::data->GetMinimum(attribute), Antecedent::data->GetMaximum(attribute));
    else {
      double size = (Antecedent::data->GetMaximum(attribute) - Antecedent::data->GetMinimum(attribute)) * data->GetDecisionCAClassPartition();
      if (Random::GetRandom(0.0, 1.0) < 0.5)
	value += size*Random::GetGaussRandom(0.3);
      else
	value -= size*Random::GetGaussRandom(0.3);

      if (value < Antecedent::data->GetMinimum(attribute))
	value = Antecedent::data->GetMinimum(attribute);
      else if (value > Antecedent::data->GetMaximum(attribute))
	value = Antecedent::data->GetMaximum(attribute);
    }
  } else if (rand > data->GetDecisionCAAttributeOptimize()) {
    if (Random::GetRandom(0.0, 1.0) < data->GetDecisionCAConditionalPartition())
      cmp = (char) Random::GetRandom(0U, 2U) * 3;
    else {
      if (cmp == 1)
	cmp = 2;
      else if (cmp == 2)
	cmp = 1;
      else
	cmp = (char) Random::GetRandom(0U, 2U) + 1;
    }
  } else {
    char *mask = Antecedent::data->GetAttributeMask();
    unsigned attributes = Antecedent::data->GetNoAttributes();
    do {
      attribute = Random::GetRandom(0U, attributes);
    } while (mask[attribute] != 1);
  }
}

ostream& operator << (ostream &os, ContinuousAntecedent &na) {
  return na.Print(os);
}

ostream& ContinuousAntecedent::Print(ostream& os) {
  if (Antecedent::data->GetSyntaxMode() == 0)
    os << "$";
  os << Antecedent::data->GetAttribute(attribute);
  switch (cmp) {
  case 3: 
    if (Antecedent::data->GetSyntaxMode() < 2)
      os << " != ";
    else
      os << " <> ";
    break;
  case 2:
    os << " > ";
    break;
  case 1:
    os << " < ";
    break;
  default:
    if (Antecedent::data->GetSyntaxMode() < 2)
      os << " == ";
    else
      os << " = ";
    break;
  }
  os << value;
  return os;
}

Antecedent* ContinuousAntecedent::Copy() {
  return new ContinuousAntecedent(*this);
}

bool ContinuousAntecedent::CoverPattern(Pattern &pattern) const {
  double val = pattern[attribute];
  switch (cmp) {
  case 3: if (val != value) return true; break;
  case 2: if (val > value) return true; break;
  case 1: if (val < value) return true; break;
  default: if (val == value) return true; break;
  }
  return false;
}
