#include "NominalAntecedent.h"

NominalAntecedent::NominalAntecedent(Data *data) : Antecedent(data) {
  char *mask = Antecedent::data->GetAttributeMask();
  unsigned attributes = Antecedent::data->GetNoAttributes();
  do {
    attribute = Random::GetRandom(0U, attributes);
  } while (mask[attribute] != 0);
}

NominalAntecedent::NominalAntecedent(const NominalAntecedent &na) : Antecedent(na) {
  attribute = na.attribute;
}

NominalAntecedent::~NominalAntecedent() {
}

void NominalAntecedent::Mutate() {
  if (Random::GetRandom(0.0, 1.0) > data->GetDecisionNAAttributeVsClassOptimize()) {
    unsigned columns = Antecedent::data->GetNoColumns();
    unsigned lower = 0;
    unsigned upper = 0;
    for (unsigned i = 0; i < columns; i += 2) {
      lower = Antecedent::data->GetColumn(i);
      upper = Antecedent::data->GetColumn(i + 1);
      if ((attribute >= lower) && (attribute < upper))
	break;
    }
    unsigned prev = attribute;
    do {
      attribute = Random::GetRandom(lower, upper);
    } while (attribute == prev);
  } else {
    char *mask = Antecedent::data->GetAttributeMask();
    unsigned attributes = Antecedent::data->GetNoAttributes();
    do {
      attribute = Random::GetRandom(0U, attributes);
    } while (mask[attribute] != 0);
  }
}

ostream& operator << (ostream &os, NominalAntecedent &na) {
  return na.Print(os);
}

ostream& NominalAntecedent::Print(ostream& os) {
  if (data->GetSyntaxMode() == 0)
    os << "$" << Antecedent::data->GetAttribute(attribute) 
       << " =~ \"" << Antecedent::data->GetValue(attribute) << "\"";
  else if (data->GetSyntaxMode() == 1)
    os << Antecedent::data->GetAttribute(attribute)
       << " == \"" << Antecedent::data->GetValue(attribute) << "\"";
  else
    os << Antecedent::data->GetAttribute(attribute)
       << " = '" << Antecedent::data->GetValue(attribute) << "'";
  return os;
}

Antecedent* NominalAntecedent::Copy() {
  return new NominalAntecedent(*this);
}

bool NominalAntecedent::CoverPattern(Pattern &pattern) const {
  double val = pattern[attribute];
  return (val >= 0.5);
}
