#include "ChromoPool.h"

Data* ChromoPool::data = NULL;

ChromoPool::ChromoPool(Data *data) {
  this->data = data;
  gaussian = false;
  unsigned initialClusters = data->GetDecisionPoolNoClustersStart();
  if (initialClusters > data->GetTrainingSet()->Length())
    initialClusters = data->GetTrainingSet()->Length();
  pool = new DynamicArray<PoolData *>(initialClusters);
  DynamicArray<Pattern *> *trainingSet = data->GetTrainingSet();
  unsigned tLength = trainingSet->Length();
  DynamicArray<Pattern *> *validationSet = data->GetValidationSet();
  unsigned vLength = validationSet->Length();
  strata = new DynamicArray<Pattern *>[initialClusters];
  noStrata = initialClusters;
  overall = new DynamicArray<Pattern *>();
  noTrainingSet = 0;

  DynamicArray<Pattern *> *trainingA;
  DynamicArray<Pattern *> *validationA;

#if SHOW_GPDECISION == 1
  cout << "Populating pool";
  cout.flush();
#endif
  unsigned clusterDiv = data->GetDecisionPoolNoClustersDivision();
  if (data->GetCrossValidation())
    data->CrossValidatePatterns();
  else
    data->ShufflePatterns();
  for (unsigned k = initialClusters; k > 0; k /= clusterDiv) {
    Clusterer *clusterer = new Clusterer(data, trainingSet, validationSet, k, data->GetDecisionPoolNoClusterEpochs());
    clusterer->FirstHeuristicOptimize();
    clusterer->Finalize();
    for (unsigned i = 0; i < k; i++) {
#if SHOW_GPDECISION == 1
      cout << ".";
      cout.flush();
#endif
      trainingA = new DynamicArray<Pattern *>(tLength);
      validationA = new DynamicArray<Pattern *>(vLength);
      Cluster *cluster = &(*clusterer)[i];
      unsigned length = cluster->Length();
      for (unsigned j = 0; j < length; j++) {
	trainingA->Add((*cluster)[j]);
      }
      unsigned vlength = cluster->ValidationLength();
      for (unsigned j = 0; j < vlength; j++) {
	validationA->Add(cluster->GetVPattern(j));
      }

      if (k == initialClusters) {
	strata[i] = *trainingA;
	noTrainingSet += strata[i].Length();
      }

      if (length > 0) {	
	Clusterer *clusterer2 = new Clusterer(data, trainingA, validationA, data->GetNoClusters(), data->GetNoClusterEpochs());
	clusterer2->SecondHeuristicOptimize();
	clusterer2->Finalize();

	GAPolyCurve *polyCurve = new GAPolyCurve(data, clusterer2, validationA, NULL);
	Chromo *chromo = new Chromo(polyCurve->Optimize());
	delete polyCurve;
	
	Chromo *chr = Add(chromo, false);
	if (chr != chromo)
	  delete chromo;
	delete clusterer2;
      }
      delete trainingA;
      delete validationA;
    }
    delete clusterer;
  }
#if SHOW_GPDECISION == 1
  cout << endl;
#endif
  InitialSample();
  //for (unsigned i = 0; i < pool->Length(); i++)
  //cout << *((*pool)[i]->chromo) << " " << (*pool)[i]->usage << " " << (*pool)[i]->sinceUsed << endl;
}

ChromoPool::~ChromoPool() {
  for (unsigned i = 0; i < pool->Length(); i++)
    delete (*pool)[i];
  delete pool;
  delete [] strata;
  delete sample;
  delete overall;
}

Chromo *ChromoPool::Get() {
  unsigned rnd;
  rnd = Random::GetRandom(0U, pool->Length());
  PoolData *ret = (*pool)[rnd];
  ret->usage++;
  return ret->chromo;
}

Chromo *ChromoPool::Add(Chromo *chromo, bool use) {
  if (*chromo == Chromo(0))
    return NULL;
  
  unsigned length = pool->Length();
  bool flag = false;
  unsigned i = 0;
  for (; i < length; i++) {
    if (chromo->cmp(*((*pool)[i]->chromo))) {
      flag = true;
      break;
    }
  }
  if (flag) {
    PoolData *ret = (*pool)[i];
    //pool->Delete(i);
    //pool->Insert(ret, 0);
    if (use)
      ret->usage++;
    return ret->chromo;
  }
  
  PoolData *pd = new PoolData();
  pd->chromo = chromo;
  if (use)
    pd->usage = 1;
  else
    pd->usage = 0;
  pd->sinceUsed = 0;
  
  pool->Add(pd);
  //pool->Insert(pd, 0);
  return chromo;
}

void ChromoPool::Remove(Chromo *chromo) {
  unsigned length = pool->Length();
  for (unsigned i = 0; i < length; i++) {
    PoolData *pd = (*pool)[i];
    if (chromo == pd->chromo) {
      pd->usage--;
      return;
    }
  }
}

void ChromoPool::Maintenance() {
  unsigned i = 0;
  while (i < pool->Length()) {
    PoolData *dat = (*pool)[i];
    if ((dat->usage == 0) && (dat->sinceUsed >= data->GetDecisionPoolFragmentLifeTime())) {
      delete dat->chromo;
      delete dat;
      pool->Delete(i);
      continue;
    } else if (dat->usage == 0) {
      dat->sinceUsed++;
      i++;
      continue;
    } else {
      dat->sinceUsed = 0;
      i++;
      continue;
    }
  }

  Sort(*pool);
  i = 0;
  for (;i < pool->Length(); i++)
    if ((*pool)[i]->usage == 0)
      break;
  
  if (Random::GetRandom(0.0, 1.0) < 0.5) {
    unsigned rnd = Random::GetRandom(0U, i);
    Chromo *tmp = new Chromo(*((*pool)[rnd]->chromo));
    tmp->MutateShrink();
    Chromo *chr = Add(tmp, false);
    if (chr != tmp)
      delete tmp;
  } else {
    unsigned rnd1 = Random::GetRandom(0U, i);
    unsigned rnd2 = Random::GetRandom(0U, pool->Length());
    Chromo *tmp = (*pool)[rnd1]->chromo->Crossover((*pool)[rnd2]->chromo);
    Chromo *chr = Add(tmp, false);
    if (chr != tmp)
      delete tmp;
  }

  //for (unsigned i = 0; i < pool->Length(); i++)
  //cout << *((*pool)[i]->chromo) << " " << (*pool)[i]->usage << " " << (*pool)[i]->sinceUsed << endl;
}

Pattern *ChromoPool::GetPattern() {
  if (noTrainingSet == 0)
    return NULL;
  double rnd = Random::GetRandom(0.0, 1.0);
  double acc = 0;
  for (unsigned i = 0; i < noStrata; i++) {
    unsigned length = strata[i].Length();
    acc += (double) length / noTrainingSet;
    if (rnd < acc) {
      Pattern *tmp = strata[i][length - 1];
      strata[i].Delete(length - 1);
      noTrainingSet--;
      return tmp;
    }
  }
  return NULL;
}

void ChromoPool::InitialSample() {
  sampleSize = data->GetDecisionInitialPercentageSampleSize() * data->GetTrainingSet()->Length();
  sample = new DynamicArray<Pattern *>((unsigned) sampleSize);
  for (unsigned i = 0; i < (unsigned) sampleSize; i++) {
    Pattern *tmp = GetPattern();
    if (tmp == NULL) {
      sampleSize = 0;
      return;
    }      
    sample->Add(tmp);
    overall->Add(tmp);
  }
  sampleSize -= (unsigned) sampleSize;
  if (sampleSize < 0)
    sampleSize = 0;
}

void ChromoPool::Resample() {
  delete sample;
  sampleSize += data->GetDecisionSampleAcceleration() * overall->Length();
  sample = new DynamicArray<Pattern *>((unsigned) sampleSize);
  for (unsigned i = 0; i < (unsigned) sampleSize; i++) {
    Pattern *tmp = GetPattern();
    if (tmp == NULL) {
      sampleSize = 0;
      return;
    }      
    sample->Add(tmp);
    overall->Add(tmp);
  }
  sampleSize -= (unsigned) sampleSize;
  if (sampleSize < 0)
    sampleSize = 0;
}

DynamicArray<Pattern *>* ChromoPool::GetTrainingSet() {
  return overall;
}

DynamicArray<Pattern *>* ChromoPool::GetSample() {
  return sample;
}

void ChromoPool::Sort(DynamicArray<PoolData *> &array) {
  QSort(array, 0, array.Length()-1);
}

void ChromoPool::QSort(DynamicArray<PoolData *> &array, int lower, int upper) {
  // first time upper = n - 1;
  // first time lower = 0;
  int i = lower;
  int j = upper;
  PoolData *pivot = array[(lower+upper)/2];
  if (lower >= upper)
    return;
  do {
    while (array[i]->usage > pivot->usage)
      i++;
    while (pivot->usage > array[j]->usage)
      j--;
    if (i <= j) {
      PoolData *tmp = array[i];
      array[i++] = array[j];
      array[j--] = tmp;
    }
  } while (i <= j);
  QSort(array, lower, j);
  QSort(array, i, upper);
}
