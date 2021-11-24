#include "GPDecision.h"

GPDecision::GPDecision(Data *data) {
#if SHOW_GPDECISION == 1
  cout << "GPDecision" << endl;
  cout << "----------" << endl;
#endif
  this->data = data;
  pool = new ChromoPool(data);
  tstd = CalculateSampleStd(pool->GetTrainingSet());
  vstd = CalculateSampleStd(data->GetValidationSet());
  noIndividuals = data->GetDecisionNoIndividuals();
  individuals = new DecisionTree*[noIndividuals];
#if SHOW_GPDECISION == 1
  cout << "Working";
  cout.flush();
#endif
  for (unsigned i = 0; i < noIndividuals; i++) {
    individuals[i] = new DecisionTree(data, pool);
    individuals[i]->Calculate(tstd, vstd);
#if SHOW_GPDECISION == 1
    cout << ".";
    cout.flush();
#endif
  }
#if SHOW_GPDECISION == 1
  cout << endl;
#endif
  Sort(individuals, noIndividuals);
  best = new DecisionTree(*(individuals[0]));
}

GPDecision::~GPDecision() {
  for (unsigned i = 0; i < noIndividuals; i++) {
    delete individuals[i];
  }
  delete [] individuals;
  delete best;
  delete pool;
}

void GPDecision::Sort(DecisionTree **array, unsigned length) {
  QSort(array, 0, length-1);
}

void GPDecision::QSort(DecisionTree **array, int lower, int upper) {
  // first time upper = n - 1;
  // first time lower = 0;
  int i = lower;
  int j = upper;
  DecisionTree *pivot = array[(lower+upper)/2];
  if (lower >= upper)
    return;
  do {
    while ((array[i]->GetFitness() > pivot->GetFitness()) ||
	   ((array[i]->GetFitness() == pivot->GetFitness()) &&
	    (array[i]->GetNoNodes() < pivot->GetNoNodes())))
      i++;
    while ((pivot->GetFitness() > array[j]->GetFitness()) ||
	   ((pivot->GetFitness() == array[j]->GetFitness()) &&
	    (pivot->GetNoNodes() < array[j]->GetNoNodes())))
      j--;
    if (i <= j) {
      DecisionTree  *tmp = array[i];
      array[i++] = array[j];
      array[j--] = tmp;
    }
  } while (i <= j);
  QSort(array, lower, j);
  QSort(array, i, upper);
}

double GPDecision::CalculateSampleStd(DynamicArray<Pattern *> *patterns) {
  unsigned length = data->GetNoAttributes();
  char * mask = data->GetAttributeMask();
  unsigned classification = length - 1;
  for (unsigned i = 0; i < length; i++)
    if (mask[i] == 3) {
      classification = i;
      break;
    }
  length = patterns->Length();
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
  return (sumsq - sum*sum / (double) length);
}

DecisionTree* GPDecision::Optimize() {
  unsigned noGenerations = data->GetDecisionNoGenerations();
  unsigned elite = (unsigned) (data->GetDecisionElite() * noIndividuals);
  unsigned crossover = (unsigned) (data->GetDecisionCrossoverRate() * noIndividuals);
  double mutation = data->GetDecisionMutationRateInitial();
  double mutationInc = data->GetDecisionMutationRateIncrement();
  double mutationMin = data->GetDecisionMutationRateInitial();
  double mutationMax = data->GetDecisionMutationRateMax();
  double lastFitness = best->GetFitness();
#if SHOW_GPDECISION == 1
  cout << 0 << " FIT: " << individuals[0]->GetFitness() << " MSE: " << individuals[0]->GetTMSE() << " BFIT: " << best->GetFitness() << " BMSE: " << best->GetTMSE() << " MR: " << mutation << " S: " << pool->GetTrainingSet()->Length() << endl;
#endif
  for (unsigned n = 1; n < noGenerations; n++) {
    if (lastFitness >= individuals[0]->GetFitness()) {
      mutation += mutationInc;
      if (mutation > mutationMax)
	mutation = mutationMin;
    } else {
      mutation = mutationMin;
    }

    DecisionTree **tmp = new DecisionTree*[noIndividuals];
#if SHOW_GPDECISION == 1
    cout << "Working";
    cout.flush();
#endif

    pool->Resample();
    tstd = CalculateSampleStd(pool->GetTrainingSet());
    DynamicArray<Pattern *> *tCover = pool->GetSample();
    for (unsigned i = 0; i < crossover; i++) {
      individuals[i]->AddCover(tCover);
    }
    best->AddCover(tCover);
    best->Calculate(tstd, vstd);
    lastFitness = individuals[0]->GetFitness();

    for (unsigned i = 0; i < elite; i++) {
      tmp[i] = new DecisionTree(*(individuals[i]));
      tmp[i]->Calculate(tstd, vstd);
#if SHOW_GPDECISION == 1
      cout << ".";
      cout.flush();
#endif
    }

    for (unsigned i = crossover; i < noIndividuals; i++) {
      delete individuals[i];
    }
    pool->Maintenance();

    for (unsigned i = elite; i < noIndividuals; i++) {
      unsigned r1 = Random::GetRandom(0U, crossover);
      unsigned r2 = Random::GetRandom(0U, crossover);
      tmp[i] = individuals[r1]->Crossover(*(individuals[r2]));
      if (Random::GetRandom(0.0, 1.0) < mutation) {
	tmp[i]->Mutate();
      }
      tmp[i]->Calculate(tstd, vstd);
#if SHOW_GPDECISION == 1
      cout << ".";
      cout.flush();
#endif
    }
#if SHOW_GPDECISION == 1
    cout << endl;
#endif
    
    for (unsigned i = 0; i < crossover; i++) {
      delete individuals[i];
    }
    delete [] individuals;
    individuals = tmp;

    Sort(individuals, noIndividuals);
    
    if ((individuals[0]->GetFitness() > best->GetFitness()) ||
	((individuals[0]->GetFitness() == best->GetFitness()) &&
	 (individuals[0]->GetNoNodes() < best->GetNoNodes()))) {
      delete best;
      best = new DecisionTree(*(individuals[0]));
    }
    
#if SHOW_GPDECISION == 1
    cout << n << " FIT: " << individuals[0]->GetFitness() << " MSE: " << individuals[0]->GetTMSE() << " BFIT: " << best->GetFitness() << " BMSE: " << best->GetTMSE() << " MR: " << mutation << " S: " << pool->GetTrainingSet()->Length() << endl;
#endif
  }

  best->SetCovers(data->GetTrainingSet(), data->GetValidationSet());
  tstd = CalculateSampleStd(data->GetTrainingSet());
  best->Parse();
  //best->Optimize();
  best->Calculate(tstd, vstd);
  double gstd = CalculateSampleStd(data->GetGeneralizationSet());
  best->CalculateGeneralization(data->GetGeneralizationSet(), gstd);
#if SHOW_GPDECISION == 1
  cout << *(best);
  cout << "FIT: " << best->GetFitness() << endl;
  cout << "NODES: " << best->GetNoNodes() << endl;
  double t,r,c;
  best->CalculateMiscellaneous(t, r, c);
  cout << "TERMS: " << t << endl;
  cout << "RULES: " << r << endl;
  cout << "CONDITIONS: " << c << endl;

  cout << "TMSE: " << best->GetTMSE();
  cout << " TACD: " << best->GetTACD();
  cout << " TCD: " << best->GetTCD();
  cout << " TMAE: " << best->GetTMAE() << endl;

  if (data->GetValidationSet()->Length() > 0) {
    cout << "VMSE: " << best->GetVMSE();
    cout << " VACD: " << best->GetVACD();
    cout << " VCD: " << best->GetVCD();
    cout << " VMAE: " << best->GetVMAE() << endl;
  }

  if (data->GetGeneralizationSet()->Length() > 0) {
    cout << "GMSE: " << best->GetGMSE();
    cout << " GACD: " << best->GetGACD();
    cout << " GCD: " << best->GetGCD();
    cout << " GMAE: " << best->GetGMAE() << endl;
  }
#endif
  return best;
}
