/* Class: Chromo
   By:    G. Potgieter

   A chromosome that represents an array of components i.e.

   y = C1*T1(x1) + C2 + ... etc.

   where each chromosome represents a function approximation.
*/

#include "Chromo.h"

Data *Chromo::data = NULL;

Chromo::Chromo(unsigned classification) {
  mse = 1.0e255;
  ae = 1.0e255;
  se = 1.0e255;
  cd = 0;
  acd = 0;
  complexity = 0;
  this->classification = classification;
}

Chromo::Chromo(unsigned classification, double coefficient) {
  mse = 1.0e255;
  ae = 1.0e255;
  se = 1.0e255;
  cd = 0;
  acd = 0;
  complexity = 0;
  this->classification = classification;
  Component cmp;
  cmp.SetCoefficient(coefficient);
  components.Add(cmp);
}

Chromo::Chromo(Data *_data, unsigned classification) {
  Chromo::data = _data;
  this->classification = classification;
  Random random;
  mse = 1.0e255;
  ae = 1.0e255;
  se = 1.0e255;
  cd = 0;
  acd = 0;
  complexity = 0;
  unsigned maxComponents = data->GetFuncMaxComponents();
  unsigned polynomialOrder = data->GetPolynomialOrder();
  unsigned noAttributes = data->GetNoAttributes();
  double ran = random.GetRandom(0.0, 1.0);
  char *mask = data->GetAttributeMask();

  if (ran < 0.1) {
    unsigned char j = random.GetRandom(1U, polynomialOrder + 1U);
    for (unsigned i = 0; ((i < maxComponents - 1U) && (i < noAttributes - 1U)); i++) {
      if (mask[i] == 1) {
	Component t(i, j);
	components.InsertUniqueSorted(t);
      }
    }
    Component u;
    components.InsertUniqueSorted(u);
  } else {
    unsigned noItems = random.GetRandom(1U, maxComponents);
    for (unsigned i = 0; (i < noItems); i++)
      MutateExpand();
  }
}

Chromo::Chromo(const Chromo &chromo) {
  se = chromo.se;
  ae = chromo.ae;
  mse = chromo.mse;
  components = chromo.components;
  classification = chromo.classification;
  cd = chromo.cd;
  acd = chromo.acd;
  complexity = chromo.complexity;
}

Chromo::~Chromo() {
  mse = 1.0e255;
  ae = 1.0e255;
  se = 1.0e255;
  cd = 0;
  acd = 0;
}

void Chromo::MutateExpand() {
  unsigned maxComponents = data->GetFuncMaxComponents();
  if (components.Length() < maxComponents) {
    Random random;
    if (random.GetRandom(0.0, 1.0) < 0.1) {
      if (components.InsertUniqueSorted(Component()))
	return;
    }
    unsigned k;
    for (k = 0; k < 10; k++) {
      Component cmp(data);
      if (components.InsertUniqueSorted(cmp))
	break;
    }
    if (k == 10)
      components.InsertUniqueSorted(Component());
  }
}

void Chromo::MutateShrink() {
  if (components.Length() > 0) {
    //select the item to delete
    Random random;
    unsigned ran = (unsigned) random.GetRandom(0U, (unsigned) components.Length());
    components.Delete(ran); 
  }
}

void Chromo::MutateComponent() {
  if (components.Length() > 0) {
    //select the item to delete
    Random random;
    unsigned ran = (unsigned) random.GetRandom(0U, (unsigned) components.Length());
    Component com = components[ran];
    components.Delete(ran); 
    if ((com == Component()) || (random.GetRandom(0.0, 1.0) < 0.1)) {
      MutateExpand();
      return;
    } else
      com.Mutate();
    components.InsertUniqueSorted(com);
  }
}

void Chromo::Mutate() {
  Random random;
  double ran = random.GetRandom(0.0, 1.0);
  if (ran < 0.25)
    MutateExpand();
  else if (ran < 0.5)
    MutateShrink();
  else if (ran < 0.75)
    MutateComponent();
  else {
    *this = Chromo(data, classification);
  }
}

Chromo * Chromo::Crossover(Chromo *chr) {
  Random random;
  unsigned maxComponents = data->GetFuncMaxComponents();
  DynamicArray<Component> cmp;

  unsigned male = 0;
  unsigned female = 0;
  unsigned maleSize = chr->components.Length();
  unsigned femaleSize = components.Length();
  while (true) {
    if ((cmp.Length() >= maxComponents) || ((female == femaleSize) && (male == maleSize))) {
      break;
    } else if ((female == femaleSize) && (male != maleSize)) {
      if (random.GetRandom(0.0, 1.0) < 0.5)
	  cmp.InsertUniqueSorted(chr->components[male]);
      male++;
    } else if ((female != femaleSize) && (male == maleSize)) {
      if (random.GetRandom(0.0, 1.0) < 0.5)
	  cmp.InsertUniqueSorted(components[female]);
      female++;
    } else if ((female != femaleSize) && (male != maleSize) && (components[female] < chr->components[male])) {
      if (random.GetRandom(0.0, 1.0) < 0.5)
	  cmp.InsertUniqueSorted(components[female]);
      female++;
    } else if ((female != femaleSize) && (male != maleSize) && (components[female] > chr->components[male])) {
      if (random.GetRandom(0.0, 1.0) < 0.5)
	  cmp.InsertUniqueSorted(chr->components[male]);
      male++;
    } else {
      //if (random.GetRandom(0.0, 1.0) < 0.5)
	cmp.InsertUniqueSorted(components[female]);
      female++;
      male++;
    }
  }
  Chromo *c = new Chromo(classification);
  //c->data = data;
  c->components = cmp;
  return c;
}

void Chromo::CalculateComponents(const DynamicArray<Pattern *> &patternList) {
  if (components.Length() != 0) {

    unsigned noOfPatterns = patternList.Length();
    Matrix *a = new Matrix(noOfPatterns, components.Length());
    Matrix *b = new Matrix(noOfPatterns, 1);

    unsigned componentLength = components.Length();
    for (unsigned i = 0; i < noOfPatterns; i++) {
      for (unsigned k = 0; k < componentLength; k++)
	a->Insert(components[k].MatrixEvaluate(*patternList[i]));
      b->Insert((*patternList[i])[classification]);
    }
    
    Matrix *aT = a->Transpose();
    Matrix *sq = aT->Multiply(a);
    delete a;
    Matrix *st = aT->Multiply(b);
    delete b;
    delete aT;
    Matrix *f = sq->Solve(st);
    delete sq;
    delete st;
    for (unsigned k = 0; k < componentLength; k++)
      components[k].SetCoefficient(f->Retrieve());
    delete f;
  }
}

void Chromo::Calculate(const DynamicArray<Pattern *> &patternList, double std) {
  CalculateComponents(patternList);
  RemoveBad();
  CalculateFitness(patternList, std);
}

void Chromo::CalculateFitness(const DynamicArray<Pattern *> &patternList, double std) {
  unsigned componentLength = components.Length();
  if (componentLength > 0) {
    double fitness = 0.0;
    double predicted = 0.0;
    ae = 0.0;
    unsigned noPatterns = patternList.Length();

    complexity = 0;
    for (unsigned i = 0; i < componentLength; i++)
      complexity += components[i].Complexity();

    double tmp;
    for (unsigned i = 0; i < noPatterns; i++) {
      predicted = 0;
      for (unsigned k = 0; k < componentLength; k++)
	predicted += components[k].Evaluate(*patternList[i]);
      tmp = ((*patternList[i])[classification] - predicted);
      fitness += tmp*tmp;
      ae += (tmp >= 0)?tmp:-tmp;
    }

    if (noPatterns == 0) {
      se = 1.0e255;
      mse = 1.0e255;
      ae = 1.0e255;
      cd = 0;
      acd = 0;
      return;
    }
    se = fitness;
    mse = fitness / noPatterns;
    
    if (std <= 0.0) {
      cd = 0.0;
      acd = 0.0;
      return;
    }
    cd = 1.0 - se / std;
    int den = noPatterns - complexity;
    if (den <= 0) {
      acd = 0.0;
      return;
    }

    acd = 1.0 - (double) (noPatterns - 1.0) / (double) den * se / std;
    if (acd <= 0.0)
      acd = 0.0;
    if (acd >= 1.0)
      acd = 1.0;
  } else {
    se = 1.0e255;
    mse = 1.0e255;
    ae = 1.0e255;
    cd = 0.0;
    acd = 0.0;
  }
}

double Chromo::Fitness() const {
  return acd;
}

double Chromo::MSE() const {
  return mse;
}

double Chromo::AE() const {
  return ae;
}

double Chromo::SSE() const {
  return se;
}

double Chromo::CD() const {
  return cd;
}

double Chromo::ACD() const {
  return acd;
}

unsigned Chromo::Complexity() const {
  return complexity;
}

unsigned Chromo::Terms() const {
  return components.Length();
}

unsigned Chromo::GetClassification() {
  return classification;
}

ostream& operator << (ostream &os, Chromo &chromo) {
  if (Chromo::data->GetSyntaxMode() == 0)
    os << "$";
  string clas = Chromo::data->GetAttribute(chromo.classification);
  string nclas = clas.substr(3, clas.length() - 3);
  os << nclas;
  if (Chromo::data->GetSyntaxMode() < 2)
    os << " = ";
  else
    os << " := ";
  for (unsigned i = 0; i < chromo.components.Length(); i++) {
    if ((chromo.components[i].GetCoefficient() >= 0) && (i != 0))
      os << "+";
    os << chromo.components[i];
  }
  return os;
}

void Chromo::RemoveBad() {
  unsigned length = components.Length();
  if (length > 0) {
    double cutoff = data->GetFuncCutOff();
    for (unsigned i = length - 1; i > 0; i--) {
      if ((components.Length() > 1) && (components[i].GetCoefficient() < cutoff) && (components[i].GetCoefficient() > -cutoff))
	components.Delete(i);
    }
    if (components.Length() > 1) {
      if ((components[0].GetCoefficient() < cutoff) && (components[0].GetCoefficient() > -cutoff))
	components.Delete(0);
    }
  }
}

bool Chromo::operator == (const Chromo &chromo) const {
  unsigned length = components.Length();
  if (length != chromo.components.Length())
    return false;
  for (unsigned i = 0; i < length; i++) {
    if (components[i] != chromo.components[i])
      return false;
  }
  return true;
} 

bool Chromo::cmp(const Chromo &chromo) const {
  unsigned length = components.Length();
  if (length != chromo.components.Length())
    return false;
  for (unsigned i = 0; i < length; i++) {
    if (components[i] != chromo.components[i])
      return false;
    if (components[i].GetCoefficient() != chromo.components[i].GetCoefficient())
      return false;
  }
  return true;
} 

bool Chromo::operator != (const Chromo &chromo) const {
  return (*this == chromo);
} 

Component& Chromo::ChooseComponent() const {
  Random random;
  unsigned choice = random.GetRandom(0U, components.Length());
  return components[choice];
}

