#include "GAPolyCurve.h"
#define HIGHSCORES 10

GAPolyCurve::GAPolyCurve(Data *data, Clusterer *strata, DynamicArray<Pattern *> *validationSet, Chromo *seed) {
#if SHOW_GAPOLYCURVE == 1
  cout << "GAPolyCurve" << endl;
  cout << "-----------" << endl;
#endif
  this->data = data;
  this->strata = strata;
  this->validationSet = validationSet;
  sample = NULL;

  unsigned noIndividuals = data->GetNoFuncIndividuals();
  unsigned noAttributes = data->GetNoAttributes();
  classification = noAttributes - 1U;
  char *mask = data->GetAttributeMask();
  for (unsigned k = 0; k < noAttributes; k++)
    if (mask[k] == 3) {
      classification = k;
      break;
    }

  noTrainingPatterns = 0;
  for (unsigned i = 0; i < strata->Length(); i++) {
    noTrainingPatterns += (*strata)[i].Length();
  }

  if (noTrainingPatterns < data->GetFuncMaxComponents())
    maximumComponents = noTrainingPatterns - 1;
  else
    maximumComponents = data->GetFuncMaxComponents();

  if (noTrainingPatterns < 100) {
    sampleSize = noTrainingPatterns;
  } else if ((unsigned) (noTrainingPatterns*data->GetFuncPercentageSampleSize()) < 100) {
    sampleSize = 100;
  } else {
    sampleSize = (unsigned) (noTrainingPatterns*data->GetFuncPercentageSampleSize());
  }

  PickPatterns();

  individuals = new Chromo*[noIndividuals];
  highScores = new Chromo*[HIGHSCORES];
  unsigned elite = (unsigned) (data->GetFuncElite() * noIndividuals);
  SampleStd();
  for (unsigned i = 0; i < noIndividuals; i++) {
    if ((i == 0) && (seed != NULL))
      individuals[i] = new Chromo(*seed);
    else if ((i < elite) && (seed != NULL)) {
      individuals[i] = new Chromo(*seed);
      individuals[i]->Mutate();
    } else
      individuals[i] = new Chromo(data, classification);
    individuals[i]->Calculate(*sample, std);
  }

  Sort(individuals, noIndividuals);
  for (unsigned i = 0; i < HIGHSCORES; i++)
    highScores[i] = new Chromo(classification);
  for (unsigned i = 0; i < noIndividuals; i++)
    ManageHighScores(individuals[i]);
}

GAPolyCurve::~GAPolyCurve() {
  unsigned noIndividuals = data->GetNoFuncIndividuals();
  for (unsigned i = 0; i < noIndividuals; i++) {
    delete individuals[i];
  }
  delete [] individuals;
  for (unsigned i = 0; i < HIGHSCORES; i++) {
    delete highScores[i];
  }
  delete [] highScores;
  if (sample != NULL)
    delete sample;
}

void GAPolyCurve::ManageHighScores(Chromo *chromo) {
  for (unsigned i = 0; i < HIGHSCORES; i++) {
    if (*(highScores[i]) == *chromo) {
      if (chromo->Fitness() > highScores[i]->Fitness()) {
	delete highScores[i];
	highScores[i] = new Chromo(*chromo);
      }
      return;
    }
  }
  for (unsigned i = 0; i < HIGHSCORES; i++) {
    if (chromo->Fitness() > highScores[i]->Fitness()) {
      delete highScores[HIGHSCORES - 1];
      for (unsigned j = HIGHSCORES - 1; j > i; j--)
	highScores[j] = highScores[j - 1];
      highScores[i] = new Chromo(*chromo);
      return;
    }
  }
}

void GAPolyCurve::Sort(Chromo **array, unsigned length) {
  QSort(array, 0, length-1);
}

void GAPolyCurve::QSort(Chromo **array, int lower, int upper) {
  // first time upper = n - 1;
  // first time lower = 0;
  int i = lower;
  int j = upper;
  Chromo *pivot = array[(lower+upper)/2];
  if (lower >= upper)
    return;
  do {
    while (array[i]->Fitness() > pivot->Fitness())
      i++;
    while (pivot->Fitness() > array[j]->Fitness())
      j--;
    if (i <= j) {
      Chromo  *tmp = array[i];
      array[i++] = array[j];
      array[j--] = tmp;
    }
  } while (i <= j);
  QSort(array, lower, j);
  QSort(array, i, upper);
}

void GAPolyCurve::PickPatterns() {
  if (sample != NULL)
    delete sample;
  sample = new DynamicArray<Pattern *>(sampleSize);
  unsigned noClusters = strata->Length();
  if (sampleSize == noTrainingPatterns) {
    for (unsigned i = 0; i < noClusters; i++) {
      Cluster *patternList = &(*strata)[i];
      unsigned size = patternList->Length();
      for (unsigned j = 0; j < size; j++) {
	sample->Add((*patternList)[j]);
      }
    }
  } else {
    for (unsigned i = 0; i < noClusters; i++) {
      Cluster *patternList = &(*strata)[i];
      unsigned stratusSize = (unsigned) rint((double) patternList->Length() / (double) noTrainingPatterns * sampleSize);
      for (unsigned j = 0; j < stratusSize; j++) {
	unsigned rnd = Random::GetRandom(0U, stratusSize);
	sample->Add((*patternList)[rnd]);
      }
    }
  }
}

void GAPolyCurve::TrainingPatterns() {
  if (sample != NULL)
    delete sample;
  unsigned noClusters = strata->Length();
  sample = new DynamicArray<Pattern *>(noTrainingPatterns);
  for (unsigned i = 0; i < noClusters; i++) {
    Cluster *patternList = &(*strata)[i];
    unsigned size = patternList->Length();
    for (unsigned j = 0; j < size; j++) {
      sample->Add((*patternList)[j]);
    }
  }
}

void GAPolyCurve::ValidationPatterns() {
  if (sample != NULL)
    delete sample;
  sample = new DynamicArray<Pattern *>(*validationSet);
}

void GAPolyCurve::GeneralizationPatterns() {
  if (sample != NULL)
    delete sample;
  sample = new DynamicArray<Pattern *>(*(data->GetGeneralizationSet()));
}

void GAPolyCurve::SampleStd() {
  unsigned length = sample->Length();
  double sum = 0;
  double sumsq = 0;
  double tmp = 0;
  for (unsigned i = 0; i < length; i++) {
    tmp = (*(*sample)[i])[classification];
    sum += tmp;
    sumsq += tmp*tmp;
  }
  if (length == 0)
    std = 0;
  else
    std = sumsq - sum*sum / (double) length;
}

Chromo& GAPolyCurve::Optimize(
#if SHOW_GAPOLYCURVE == 1
double *outArray
#endif
) {
  Random random;
  unsigned noGenerations = data->GetNoFuncGenerations();
  unsigned noIndividuals = data->GetNoFuncIndividuals();
  unsigned crossover = (unsigned) (data->GetFuncCrossoverRate() * noIndividuals);
  unsigned elite = (unsigned) (data->GetFuncElite() * noIndividuals);
  Chromo **tmpPopulation;

#if SHOW_GAPOLYCURVE == 1
  cout << 0U << " FIT: " << individuals[0]->Fitness() << " MSE: " << individuals[0]->MSE() 
       << " BFIT: " << highScores[0]->Fitness() << " BMSE: " << highScores[0]->MSE() << endl;
#endif

  for (unsigned n = 1; n < noGenerations; n++) {

    PickPatterns();
    tmpPopulation = new Chromo*[noIndividuals];

    SampleStd();
    for (unsigned i = 0; i < elite; i++) {
      tmpPopulation[i] = new Chromo(*individuals[i]);
      tmpPopulation[i]->Calculate(*sample, std);
    }

    for (unsigned i = elite; i < noIndividuals; i++) {
      unsigned first = random.GetRandom(0U, crossover);
      unsigned second = random.GetRandom(0U, crossover);
      tmpPopulation[i] = individuals[first]->Crossover(individuals[second]);
      double rnd = random.GetRandom(0.0, 1.0);
      if (rnd > data->GetFuncMutationRate()) {
        tmpPopulation[i]->Mutate();
      }
      tmpPopulation[i]->Calculate(*sample, std);
    }

    for (unsigned i = 0; i < noIndividuals; i++) {
      delete individuals[i];
    }
    delete [] individuals;
    individuals = tmpPopulation;

    Sort(individuals, noIndividuals);

    for (unsigned i = 0; i < crossover; i++)
      ManageHighScores(individuals[i]);

#if SHOW_GAPOLYCURVE == 1
    cout << n << " FIT: " << individuals[0]->Fitness() << " MSE: " << individuals[0]->MSE()
	 << " BFIT: " << highScores[0]->Fitness() << " BMSE: " << highScores[0]->MSE() << endl;
#endif
  }

  if (validationSet->Length() != 0) {
    ValidationPatterns();
    SampleStd();
    for (unsigned i = 0; i < HIGHSCORES; i++)
      highScores[i]->Calculate(*sample, std);
    Sort(highScores, HIGHSCORES);
  } else {
    TrainingPatterns();
    SampleStd();
    for (unsigned i = 0; i < HIGHSCORES; i++)
      highScores[i]->Calculate(*sample, std);
    Sort(highScores, HIGHSCORES);
  }

#if SHOW_GAPOLYCURVE == 1
  cout << *highScores[0] << endl;
  TrainingPatterns();
  SampleStd();
  highScores[0]->CalculateFitness(*sample, std);

  outArray[0] = highScores[0]->Fitness();
  outArray[1] = highScores[0]->MSE();
  outArray[2] = highScores[0]->ACD();
  outArray[3] = highScores[0]->CD();

  cout << "TFIT: " << outArray[0] << " TMSE: " << outArray[1];
  cout << " TACD: " << outArray[2] << " TCD: " << outArray[3] << endl;

  if (validationSet->Length() != 0) {
    ValidationPatterns();
    SampleStd();
    highScores[0]->CalculateFitness(*sample, std);

    outArray[4] = highScores[0]->Fitness();
    outArray[5] = highScores[0]->MSE();
    outArray[6] = highScores[0]->ACD();
    outArray[7] = highScores[0]->CD();

    cout << "VFIT: " << outArray[4] << " VMSE: " << outArray[5];
    cout << " VACD: " << outArray[6] << " VCD: " << outArray[7] << endl;
  }

  if (data->GetGeneralizationSet()->Length() != 0) {
    GeneralizationPatterns();
    SampleStd();
    highScores[0]->CalculateFitness(*sample, std);

    outArray[8] = highScores[0]->Fitness();
    outArray[9] = highScores[0]->MSE();
    outArray[10] = highScores[0]->ACD();
    outArray[11] = highScores[0]->CD();

    cout << "GFIT: " << outArray[8] << " GMSE: " << outArray[9];
    cout << " GACD: " << outArray[10] << " GCD: " << outArray[11] << endl;
  }
  outArray[12] = highScores[0]->Complexity();
  outArray[13] = highScores[0]->Terms();
#endif
  return *highScores[0];
}


