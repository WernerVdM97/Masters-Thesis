#include "ContinuousConsequent.h"

ChromoPool *ContinuousConsequent::chromoPool = NULL;

ContinuousConsequent::ContinuousConsequent(Data *data, ChromoPool *chromoPool,DynamicArray<Pattern *> *tCover, DynamicArray<Pattern *> *vCover) : Consequent(data, tCover, vCover) {
  if (chromoPool == NULL)
    throw new Exception("ChromoPool NULL in Continuous Consequent, Constructor");
  ContinuousConsequent::chromoPool = chromoPool;
  chromo = chromoPool->Get();
  complexity = chromo->Complexity();
  terms = chromo->Terms();
}

ContinuousConsequent::ContinuousConsequent(const ContinuousConsequent &cc) : Consequent(cc) {
  chromo = chromoPool->Add(cc.chromo, true);
}

ContinuousConsequent::~ContinuousConsequent() {
  chromoPool->Remove(chromo);
}

void ContinuousConsequent::Mutate() {
  if (trainingCover->Length() > 0) {
    if (Random::GetRandom(0.0, 1.0) < data->GetDecisionReoptimizeVsSelectLeaf()) {
      Optimize();
    } else {
      chromoPool->Remove(chromo);
      chromo = chromoPool->Get();
      complexity = chromo->Complexity();
      terms = chromo->Terms();
    }
  } else {
    chromoPool->Remove(chromo);
    chromo = chromoPool->Get();
    complexity = chromo->Complexity();
    terms = chromo->Terms();
  }
  tsse = -1;
  tmse = -1;
  tae = -1;
  vsse = -1;
  vmse = -1;
  vae = -1;
}

ostream& operator << (ostream &os, ContinuousConsequent &cc) {
  return cc.Print(os);
}

ostream& ContinuousConsequent::Print(ostream& os) {
  if (chromo != NULL)
    os << *chromo;
  if (data->GetSyntaxMode() < 2)
    os << ";";
  os << " ";
  if (data->GetSyntaxMode() == 0)
    os << "#";
  else if (data->GetSyntaxMode() == 1)
    os << "//";
  else
    os << "{";
  os << "(" << trainingCover->Length() << ", " << validationCover->Length() << ") <" << GetTMSE() << ", " << GetVMSE() << ">";
  if (data->GetSyntaxMode() == 2)
    os << "}";
  return os;
}

Consequent* ContinuousConsequent::Copy() {
  return new ContinuousConsequent(*this);
}

void ContinuousConsequent::Optimize() {
  if (validationCover == NULL)
    throw new Exception("validationCover NULL in Continuous Consequent, Optimize");

  unsigned length = trainingCover->Length();
  for (unsigned i = 0; i < length; i++) {
    unsigned rnd = Random::GetRandom(0U, length);
    unsigned rnd2 = Random::GetRandom(0U, length);
    Pattern *tmp = (*trainingCover)[rnd];
    (*trainingCover)[rnd] = (*trainingCover)[rnd2];
    (*trainingCover)[rnd2] = tmp;
  }

  CalculateFitness();

  Clusterer clusterer(data, trainingCover, validationCover, data->GetNoClusters(), data->GetNoClusterEpochs());
  clusterer.SecondHeuristicOptimize();
  clusterer.Finalize();

  GAPolyCurve polyCurve(data, &clusterer, validationCover, this->chromo);
  chromoPool->Remove(this->chromo);
  Chromo *chromo = new Chromo(polyCurve.Optimize());
  this->chromo = chromoPool->Add(chromo, true);
  if (this->chromo == NULL) {
    delete chromo;
    this->chromo = chromoPool->Get();
  } else {
    if (this->chromo != chromo)
      delete chromo;
  }
  complexity = this->chromo->Complexity();
  terms = this->chromo->Terms();
}

double ContinuousConsequent::CalculateStd(DynamicArray<Pattern *> *patterns, unsigned classification) {
  unsigned length = patterns->Length();
  double sum = 0;
  double sumsq = 0;
  double tmp = 0;
  for (unsigned i = 0; i < length; i++) {
    tmp = (*(*patterns)[i])[classification];
    sum += tmp;
    sumsq += tmp*tmp;
  }

  if (length == 0)
    return 0.0;
  return sumsq - sum*sum / (double) length;
}

void ContinuousConsequent::CalculateFitness() {
  unsigned tCover = trainingCover->Length();
  unsigned vCover = validationCover->Length();
  if ((tCover == 0) && (vCover == 0)) {
    tsse = -1;
    tmse = -1;
    tae = -1;
    vsse = -1;
    vmse = -1;
    vae = -1;
    return;
  }

  if (chromo == NULL)
    throw new Exception("chromo NULL in Continuous Consequent, Calculate Fitness");

  unsigned classification = chromo->GetClassification();

  if ((tsse < 0) && (tCover > 0)) {
    chromo->CalculateFitness(*trainingCover, CalculateStd(trainingCover, classification));
    tsse = chromo->SSE();
    tmse = chromo->MSE();
    tae = chromo->AE();
  }
  if ((vsse < 0) && (vCover > 0)) {
    chromo->CalculateFitness(*validationCover, CalculateStd(validationCover, classification));
    vsse = chromo->SSE();
    vmse = chromo->MSE();
    vae = chromo->AE();
  }
}

bool ContinuousConsequent::InUse(Chromo *chr) {
  if (chromo == chr)
    return true;
  return false;
}

void ContinuousConsequent::CalculateGeneralization(DynamicArray<Pattern *> *gCover, double &sse, double &ae) {
  if (gCover->Length() > 0) {
    chromo->CalculateFitness(*gCover, 0);
    sse += chromo->SSE();
    ae += chromo->AE();
    return;
  }
  return;
}

