#include <cstdlib>
#include <ctime>
#include <fstream>
#include "baseClasses/Data.h"
#include "k-means/Clusterer.h"
#include "gapolycurve/GAPolyCurve.h"

char *simfilename;
char *bestfilename;
char *resultfilename;

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
}

double storage[15][2] = {{0,0},{0,0},{0,0},{0,0},{0,0},
			 {0,0},{0,0},{0,0},{0,0},{0,0},
			 {0,0},{0,0},{0,0},{0,0},{0,0}};

void outputSimulation(unsigned n) {
  ofstream file(simfilename);
  file << n << " ";
  for (unsigned i = 0; i < 15; i++) {
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

  for (unsigned i = 0; i < 15; i++) {
    file >> storage[i][0] >> storage[i][1];
  }

  file.close();
  return n + 1;
}

void outputBest(double * storage, Chromo & best) {
  ofstream file(bestfilename, ios::app);
  file << best << endl;
  file << "TFIT: " << storage[0];
  file << " VFIT: " << storage[4];
  file << " GFIT: " << storage[8] << endl;

  file << "TMSE: " << storage[1];
  file << " VMSE: " << storage[5];
  file << " GMSE: " << storage[9] << endl;

  file << "TACD: " << storage[2];
  file << " VACD: " << storage[6];
  file << " GACD: " << storage[10] << endl;

  file << "TCD: " << storage[3];
  file << " VCD: " << storage[7];
  file << " GCD: " << storage[11] << endl;

  file << "TERMS: " << storage[12] << endl;
  file << "COMPLEXITY: " << storage[13] << endl;

  file.close();
}

void outputResults(unsigned max, ostream &file) {
  file << "TFIT: " << storage[0][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[0][1] - storage[0][0]*storage[0][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "TMSE: " << storage[1][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[1][1] - storage[1][0]*storage[1][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "TACD: " << storage[2][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[2][1] - storage[2][0]*storage[2][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "TCD: " << storage[3][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[3][1] - storage[3][0]*storage[3][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "VFIT: " << storage[4][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[4][1] - storage[4][0]*storage[4][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "VMSE: " << storage[5][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[5][1] - storage[5][0]*storage[5][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "VACD: " << storage[6][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[6][1] - storage[6][0]*storage[6][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "VCD: " << storage[7][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[7][1] - storage[7][0]*storage[7][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "GFIT: " << storage[8][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[8][1] - storage[8][0]*storage[8][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "GMSE: " << storage[9][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[9][1] - storage[9][0]*storage[9][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "GACD: " << storage[10][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[10][1] - storage[10][0]*storage[10][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "GCD: " << storage[11][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[11][1] - storage[11][0]*storage[11][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "COMPLEXITY: " << storage[12][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[12][1] - storage[12][0]*storage[12][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "TERMS: " << storage[13][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[13][1] - storage[13][0]*storage[13][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "TIME: " << storage[14][0] / (double) max;
  if (max > 1)
    file << " " << sqrt((storage[14][1] - storage[14][0]*storage[14][0] / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;
}

int main(int argc, char **argv) {
  double outArray[14];
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
      GAPolyCurve polyCurve(data, &clusterer, data->GetValidationSet(), NULL);
      Chromo chr = polyCurve.Optimize(outArray);

      clock_t end = clock();
      time_t endBig = time(NULL);
      if (endBig - startBig > 100)
	cout << "Time: " << ((double) (endBig - startBig)) << "s" << endl;
      else
	cout << "Time: " << ((double) (end - start)) / CLOCKS_PER_SEC << "s" << endl;

      for (int i = 0; i < 14; i++) {
	storage[i][0] += outArray[i];
	storage[i][1] += outArray[i]*outArray[i];
      }

      if (endBig - startBig > 100) {
	  storage[14][0] += ((double) (endBig - startBig));
	  storage[14][1] += ((double) (endBig - startBig)) * ((double) (endBig - startBig));
	} else {
	  storage[14][0] += ((double) (end - start)) / CLOCKS_PER_SEC;
	  storage[14][1] += ((double) (end - start)) / CLOCKS_PER_SEC * ((double) (end - start)) / CLOCKS_PER_SEC;
	}

      outputSimulation(n);
      outputBest(outArray, chr);

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



