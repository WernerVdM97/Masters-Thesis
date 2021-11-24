#include "Antecedent.h"

Data *Antecedent::data = NULL;

Antecedent::Antecedent(Data * data) {
  Antecedent::data = data;
  if (Antecedent::data == NULL)
    throw new NullPointerException;
}

Antecedent::Antecedent(const Antecedent &antecedent) {  
}

Antecedent::~Antecedent() {
}

ostream& operator << (ostream &os, Antecedent &antecedent) {
  return antecedent.Print(os);
}





