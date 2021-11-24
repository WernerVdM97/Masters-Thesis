#include <cstdlib>
#include <ctime>
#include <fstream>
#include "baseClasses/Data.h"
#include "k-means/Clusterer.h"
#include "neuralnetwork/NeuralNetwork.h"

char *simfilename;
char *bestfilename;
char *resultfilename;
char *outfilename;
unsigned outlen;

void setFileNames(char *filename) {
  unsigned length = strlen(filename);
  simfilename = new char[length + 5];
  for (unsigned i = 0; i < length; i++)
    simfilename[i] = filename[i];
  simfilename[length] = '.';
  simfilename[length + 1] = 's';
  simfilename[length + 2] = 'i';
  simfilename[length + 3] = 'm';
  simfilename[length + 4] = '\0';

  bestfilename = new char[length + 6];
  for (unsigned i = 0; i < length; i++)
    bestfilename[i] = filename[i];
  bestfilename[length] = '.';
  bestfilename[length + 1] = 'b';
  bestfilename[length + 2] = 'e';
  bestfilename[length + 3] = 's';
  bestfilename[length + 4] = 't';
  bestfilename[length + 5] = '\0';

  resultfilename = new char[length + 8];
  for (unsigned i = 0; i < length; i++)
    resultfilename[i] = filename[i];
  resultfilename[length] = '.';
  resultfilename[length + 1] = 'r';
  resultfilename[length + 2] = 'e';
  resultfilename[length + 3] = 's';
  resultfilename[length + 4] = 'u';
  resultfilename[length + 5] = 'l';
  resultfilename[length + 6] = 't';
  resultfilename[length + 7] = '\0';

  outfilename = new char[length + 9];
  for (unsigned i = 0; i < length; i++)
    outfilename[i] = filename[i];
  outfilename[length] = '.';
  outfilename[length + 1] = 'o';
  outfilename[length + 2] = 'u';
  outfilename[length + 3] = 't';
  outfilename[length + 4] = '.';
  outfilename[length + 5] = '0';
  outfilename[length + 6] = '0';
  outfilename[length + 7] = '0';
  outfilename[length + 8] = '\0';
  outlen = length + 9;
}

double storage[4][2] = {{0,0},{0,0},{0,0},{0,0}};

void outputSimulation(unsigned n) {
  ofstream file(simfilename);
  file << n << " ";
  for (unsigned i = 0; i < 4; i++) {
    file << storage[i][0] << " " << storage[i][1] << " ";
  }

  file.close();
}

unsigned inputSimulation() {
  ifstream file(simfilename);
  if (file == NULL)
    return 0;
  unsigned n;
  file >> n;

  for (unsigned i = 0; i < 4; i++) {
    file >> storage[i][0] >> storage[i][1];
  }

  file.close();
  return n + 1;
}

void outputBest(double * storage) {
  ofstream file(bestfilename, ios::app);
  file << "TMSE: " << storage[0];
  file << " VMSE " << storage[1];
  file << " GMSE: " << storage[2] << endl;
  file << "TIME: " << storage[3] << endl;

  file.close();
}

void outputResults(unsigned max, ostream &file) {
  file << "TMSE: " << storage[0][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[0][1] - storage[0][0]*storage[0][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "VMSE: " << storage[1][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[1][1] - storage[1][0]*storage[1][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "GMSE: " << storage[2][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[2][1] - storage[2][0]*storage[2][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "TIME: " << storage[3][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[3][1] - storage[3][0]*storage[3][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;
}



int main(int argc, char **argv) {
  double outArray[4];
  try {
    if (argc < 2) {
      cout << "Usage: " << argv[0] << " <Data_name> [simulations]" << endl;
      exit(-1);
    }
    Random::Seed();
    setFileNames(argv[1]);
    Data *data = new Data(argv[1]);

    unsigned length = 1;
    if (argc == 3)
      length = atoi(argv[2]);

    unsigned n = (length > 1)?inputSimulation():0;
    if (data->GetCrossValidation())
      for (unsigned i = 0; i < n; i++)
	data->CrossValidatePatterns();
    else
      data->ShufflePatterns();

    for (; n < length; n++) {
      outfilename[outlen - 2] = (char) ((n % 10) + '0');
      outfilename[outlen - 3] = (char) ((n % 100 / 10) + '0');
      outfilename[outlen - 4] = (char) ((n % 1000 / 100) + '0');

      if (length > 1)
	cout << "Simulation: " << (n + 1) << " of " << length << endl;
      clock_t start = clock();
      time_t startBig = time(NULL);
      Clusterer clusterer(data, data->GetTrainingSet(), data->GetValidationSet(), data->GetNoClusters(), data->GetNoClusterEpochs());
      
#if CLUSTER_HEUR == 0
      clusterer.NoHeuristicOptimize();
#elif CLUSTER_HEUR == 1
      clusterer.FirstHeuristicOptimize();
#elif CLUSTER_HEUR == 2
      clusterer.SecondHeuristicOptimize();
#endif
      clusterer.Finalize();
      NeuralNetwork neural(data, &clusterer);
      neural.Optimize(outArray, outfilename);

      clock_t end = clock();
      time_t endBig = time(NULL);
      if (endBig - startBig > 100)
	cout << "Time: " << ((double) (endBig - startBig)) << "s" << endl;
      else
	cout << "Time: " << ((double) (end - start)) / CLOCKS_PER_SEC << "s" << endl;

      for (int i = 0; i < 3; i++) {
	storage[i][0] += outArray[i];
	storage[i][1] += outArray[i]*outArray[i];
      }

      if (endBig - startBig > 100) {
	  storage[3][0] += ((double) (endBig - startBig));
	  storage[3][1] += ((double) (endBig - startBig)) * ((double) (endBig - startBig));
	} else {
	  storage[3][0] += ((double) (end - start)) / CLOCKS_PER_SEC;
	  storage[3][1] += ((double) (end - start)) / CLOCKS_PER_SEC * ((double) (end - start)) / CLOCKS_PER_SEC;
	}

      outputSimulation(n);
      outputBest(outArray);
      
      if (data->GetCrossValidation())
	data->CrossValidatePatterns();
      else
	data->ShufflePatterns();
    }
    if (length > 1) {
      ofstream file(resultfilename);
      outputResults(length, cout);
      outputResults(length, file);
      file.close();
    }
    //delete data;
    } catch (Exception *exception) {
    cout << *exception << endl;
    delete exception;
    exit(-1);
  }
}



