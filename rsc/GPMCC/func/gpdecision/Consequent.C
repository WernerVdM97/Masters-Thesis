#include "Consequent.h"

Data *Consequent::data = NULL;

Consequent::Consequent(Data *data, DynamicArray<Pattern *> *t, DynamicArray<Pattern *> *v) {
  if (data == NULL)
    throw new Exception("Data NULL in Consequent, Constructor");
  Consequent::data = data;
  if (t == NULL)
    throw new Exception("TrainingCover NULL in Consequent, Constructor");
  trainingCover = t;
  if (v == NULL)
    throw new Exception("ValidationCover NULL in Consequent, Constructor");
  validationCover = v;
  tsse = -1;
  tmse = -1;
  tae = -1;
  vsse = -1;
  vmse = -1;
  vae = -1;
  
  complexity = 0;
  terms = 0;
}

Consequent::Consequent(const Consequent &consequent) {
  trainingCover = new DynamicArray<Pattern *>(*(consequent.trainingCover));
  validationCover = new DynamicArray<Pattern *>(*(consequent.validationCover));
  tsse = consequent.tsse;
  tmse = consequent.tmse;
  tae = consequent.tae;
  vsse = consequent.vsse;
  vmse = consequent.vmse;
  vae = consequent.vae;

  complexity = consequent.complexity;
  terms = consequent.terms;
}

Consequent::~Consequent() {
  delete trainingCover;
  delete validationCover;
}

ostream& operator << (ostream &os, Consequent &consequent) {
  return consequent.Print(os);
}

void Consequent::SetCovers(DynamicArray<Pattern *> *tCover, DynamicArray<Pattern *> *vCover) {
  tsse = -1;
  tmse = -1;
  tae = -1;
  vsse = -1;
  vmse = -1;
  vae = -1;
  delete trainingCover;
  delete validationCover;
  if (tCover == NULL)
    throw new Exception("TrainingCover NULL in Consequent, SetCovers");
  trainingCover = tCover;
  if (vCover == NULL)
    throw new Exception("ValidationCover NULL in Consequent, SetCovers");
  validationCover = vCover;
}

void Consequent::AddCover(DynamicArray<Pattern *> *tCover) {
  tsse = -1;
  tmse = -1;
  tae = -1;
  if (tCover == NULL)
    throw new Exception("TrainingCover NULL in Consequent, AddCovers");
  unsigned length = tCover->Length();
  for (unsigned i = 0; i < length; i++)
    trainingCover->Add((*tCover)[i]);
}

unsigned Consequent::GetNoTrainingCover() {
  return trainingCover->Length();
}

unsigned Consequent::GetNoValidationCover() {
  return validationCover->Length();
}

void Consequent::GetCovers(DynamicArray<Pattern *> *tCover, DynamicArray<Pattern *> *vCover) {
  unsigned length = trainingCover->Length();
  for (unsigned i = 0; i < length; i++)
    tCover->Add((*trainingCover)[i]);
  length = validationCover->Length();
  for (unsigned i = 0; i < length; i++)
    vCover->Add((*validationCover)[i]);
}

double Consequent::GetTSSE() const {
  if (tsse < 0)
    return 0;
  return tsse;
}

double Consequent::GetTMSE() const {
  if (tmse < 0)
    return 0;
  return tmse;
}

double Consequent::GetVSSE() const {
  if (vsse < 0)
    return 0;
  return vsse;
}

double Consequent::GetVMSE() const {
  if (vmse < 0)
    return 0;
  return vmse;
}

unsigned Consequent::GetComplexity() const {
  return complexity;
}

unsigned Consequent::GetTerms() const {
  return terms;
}

double Consequent::GetTMAE() const {
  if (tae < 0)
    return 0;
  return tae / trainingCover->Length();
}

double Consequent::GetVMAE() const {
  if (vae < 0)
    return 0;
  return vae / validationCover->Length();
}

double Consequent::GetTAE() const {
  if (tae < 0)
    return 0;
  return tae;
}

double Consequent::GetVAE() const {
  if (vae < 0)
    return 0;
  return vae;
}
