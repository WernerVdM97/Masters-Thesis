/* Class: Clusterer
   By:    G. Potgieter

   A k-means optimizer with multiple built in heuristics.
*/

#include "Clusterer.h"

Clusterer::Clusterer(Data *data, DynamicArray<Pattern *> *trainingSet, DynamicArray<Pattern *> *validationSet, unsigned noClusters, unsigned noEpochs) {
#if SHOW_CLUSTERER == 1
  cout << "k-means" << endl;
  cout << "-----------" << endl;
#endif
  if (data == NULL)
    throw new NullPointerException();
  this->data = data;
  unsigned training = trainingSet->Length();
  unsigned validation = validationSet->Length();

  this->noEpochs = noEpochs;
  this->noClusters = noClusters;
  if (this->noClusters > training)
    this->noClusters = training;
  clusters = new Cluster[noClusters];

  
  unsigned start = Random::GetRandom(0U, training);
  unsigned current = 0;
  for (unsigned i = 0; i < training; i++) {
    current = (start + i) % training;
    clusters[i % noClusters].AddPattern((*trainingSet)[current]);
  }
  
  //for (unsigned i = 0; i < training; i++)
  //clusters[i % noClusters].AddPattern((*trainingSet)[i]);
  
  for (unsigned i = 0; i < validation; i++)
    clusters[i % noClusters].AddValidationPattern((*validationSet)[i]);
}

Clusterer::~Clusterer() {
  data = NULL;
  delete [] clusters;
}

void Clusterer::NoHeuristicOptimize() {
  for (unsigned n = 0; n < noEpochs; n++) {
    for (unsigned i = 0; i < noClusters; i++)
      clusters[i].TestCloser(clusters, noClusters);
  }
}

void Clusterer::FirstHeuristicOptimize() {
  double PMQE = 1000000000000;
  double MQE;
  unsigned training = 0;
  for (unsigned i = 0; i < noClusters; i++)
    training += clusters[i].Length();
  for (unsigned n = 0; n < noEpochs; n++) {
    for (unsigned i = 0; i < noClusters; i++)
      clusters[i].TestCloser(clusters, noClusters);

    MQE = 0;
    for (unsigned i = 0; i < noClusters; i++) {
      MQE += clusters[i].GetQE();
    }
    MQE /= training;
#if SHOW_CLUSTERER == 1
    cout << n << "\tMQE: " << MQE << endl;
#endif
    if (MQE > PMQE)
      break;
    PMQE = MQE;
  }
}

void Clusterer::SecondHeuristicOptimize() {
  double PMQE = 1000000000000;
  double PGMQE = 1000000000000;
  double MQE;
  double GMQE;
  unsigned training = 0;
  for (unsigned i = 0; i < noClusters; i++)
    training += clusters[i].Length();

 unsigned validation = 0;
  for (unsigned i = 0; i < noClusters; i++)
    validation += clusters[i].ValidationLength();

  for (unsigned n = 0; n < noEpochs; n++) {
    for (unsigned i = 0; i < noClusters; i++)
      clusters[i].TestCloser(clusters, noClusters);
    for (unsigned i = 0; i < noClusters; i++)
      clusters[i].TestValidationCloser(clusters, noClusters);

    MQE = 0;
    for (unsigned i = 0; i < noClusters; i++) {
      MQE += clusters[i].GetQE();
    }
    MQE /= training;

    GMQE = 0;
    for (unsigned i = 0; i < noClusters; i++) {
      GMQE += clusters[i].GetVQE();
    }
    GMQE /= validation;

#if SHOW_CLUSTERER == 1
    cout << n << "\tMQE: " << MQE << "\tGMQE: " << GMQE << endl;
#endif

    if ((MQE > PMQE) || (GMQE > PGMQE))
      break;
    PMQE = MQE;
    PGMQE = GMQE;    
  }
}

void Clusterer::Finalize() {
  for (unsigned i = 0; i < noClusters; i++)
    clusters[i].Finalize();
}

unsigned Clusterer::Length() const {
  return noClusters;
}

Cluster& Clusterer::operator [] (unsigned index) const {
  if (index > noClusters)
    throw new IndexOutOfBoundsException();
  return clusters[index];
}
