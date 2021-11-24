#include "NeuralNetwork.h"

Data *NeuralNetwork::data = NULL;

NeuralNetwork::NeuralNetwork(Data *data, Clusterer *strata) {
#if SHOW_NEURALNETWORK == 1
  cout << "NeuralNetwork" << endl;
  cout << "-----------" << endl;
#endif
  this->data = data;
  noInputs = data->GetNeuralInputUnits();
  noHiddens = data->GetNeuralHiddenUnits();
  noOutputs = data->GetNeuralOutputUnits();
  lambda = data->GetNeuralActivation();
  eta = data->GetNeuralLearningRate();
  alpha = data->GetNeuralMomentum();
  epochs = data->GetNeuralEpochs();
  sample = NULL;
  this->strata = strata;
  
  //init - input to hidden layer
  double base = 1.0 / sqrt(noInputs + 1.0);
  inputHidden = new double*[noInputs + 1];
  inputHidden_dw = new double*[noInputs + 1];
  inputHidden_dwprev = new double*[noInputs + 1];
  for (unsigned i = 0; i < noInputs + 1; i++) {
    inputHidden[i] = new double[noHiddens];
    inputHidden_dw[i] = new double[noHiddens];
    inputHidden_dwprev[i] = new double[noHiddens];
    for (unsigned h = 0; h < noHiddens; h++) {
      inputHidden[i][h] = Random::GetRandom(-base, base);
      inputHidden_dw[i][h] = 0;
      inputHidden_dwprev[i][h] = 0;
    }
  }

  //init - hidden to output layer
  base = 1.0 / sqrt(noHiddens + 1.0);
  hiddenOutput = new double*[noHiddens + 1];
  hiddenOutput_dw = new double*[noHiddens + 1];
  hiddenOutput_dwprev = new double*[noHiddens + 1];
  for (unsigned h = 0; h < noHiddens + 1; h++) {
    hiddenOutput[h] = new double[noOutputs];
    hiddenOutput_dw[h] = new double[noOutputs];
    hiddenOutput_dwprev[h] = new double[noOutputs];
    for (unsigned o = 0; o < noOutputs; o++) {
      hiddenOutput[h][o] = Random::GetRandom(-base, base);
      hiddenOutput_dw[h][o] = 0;
      hiddenOutput_dwprev[h][o] = 0;
    }
  }

  noTrainingPatterns = 0;
  for (unsigned i = 0; i < strata->Length(); i++) {
    noTrainingPatterns += (*strata)[i].Length();
  }

  if (noTrainingPatterns < 100) {
    sampleSize = noTrainingPatterns;
  } else if ((unsigned) (noTrainingPatterns * data->GetNeuralPercentageSampleSize()) < 100) {
    sampleSize = 100;
  } else {
    sampleSize = (unsigned) (noTrainingPatterns * data->GetNeuralPercentageSampleSize());
  }

  pickPatterns();
}

NeuralNetwork::~NeuralNetwork() {
  for (unsigned i = 0; i < noInputs + 1; i++) {
    delete [] inputHidden[i];
    delete [] inputHidden_dw[i];
  }
  delete [] inputHidden;
  delete [] inputHidden_dw;
  for (unsigned h = 0; h < noHiddens + 1; h++) {
    delete [] hiddenOutput[h];
    delete [] hiddenOutput_dw[h];
  }
  delete hiddenOutput;
  delete hiddenOutput_dw;
}

void NeuralNetwork::formatPattern(Pattern &pattern, 
				  const char *mask,
				  double *input) {
  unsigned a = 0;
  unsigned b = noInputs;
  unsigned length = noInputs + noOutputs;
  double minm, maxm;
  for (unsigned i = 0; i < length; i++) {
    minm = data->GetMinimum(i);
    maxm = data->GetMaximum(i);
    if ((mask[i] & 2) == 0) {
      input[a] = (pattern[i] - minm) / (maxm - minm) * 3.464101615 - 1.7320508075;
      a++;
    } else {
      input[b] = pattern[i]/*(pattern[i] - minm) / (maxm - minm) * 3.464101615 - 1.7320508075*/;
      b++;
    }
  }
}

void NeuralNetwork::forwardPropagate(double *input, 
				     double *hidden, 
				     double *output) {
  // pre: input is filled with attributes ->|<- targets

  for (unsigned h = 0; h < noHiddens; h++) {
    for (unsigned i = 0; i < noInputs; i++) {
      hidden[h] += input[i] * inputHidden[i][h];
    }
    hidden[h] += inputHidden[noInputs][h];

    // sigmoid activation in hidden layer;
    hidden[h] = 1.0 / (1.0 + exp(-lambda * hidden[h]));
  }

  for (unsigned o = 0; o < noOutputs; o++) {
    for (unsigned h = 0; h < noHiddens; h++) {
      output[o] += hidden[h] * hiddenOutput[h][o];
    }
    output[o] += hiddenOutput[noHiddens][o];

    // linear activation in output layer
    //output[o] = 1.0 / (1.0 + exp(-lambda * output[o]));
  }
}

void NeuralNetwork::clearDeltas() {
  for (unsigned i = 0; i < noInputs + 1; i++)
    for (unsigned h = 0; h < noHiddens; h++)
      inputHidden_dw[i][h] = 0;
  for (unsigned h = 0; h < noHiddens + 1; h++)
    for (unsigned o = 0; o < noOutputs; o++)
      hiddenOutput_dw[h][o] = 0;
}

double NeuralNetwork::backPropagate(const double *input, 
				    const double *hidden, 
				    const double *output) {

  double *outputError = new double[noOutputs];
  double sse = 0.0;
  for (unsigned o = 0; o < noOutputs; o++) {
    outputError[o] = input[noInputs + o] - output[o];
    sse += outputError[o] * outputError[o];
    //outputError[o] *= output[o] * (1 - output[o]);

    // correct hidden - ouput
    for (unsigned h = 0; h < noHiddens; h++) {
      hiddenOutput_dw[h][o] += eta * outputError[o] * hidden[h];
    }
    // bias
    hiddenOutput_dw[noHiddens][o] += eta * outputError[o];
  }
  sse = sse / (double) noOutputs;

  double *hiddenError = new double[noHiddens];
  for (unsigned h = 0; h < noHiddens; h++) {
    // calculate hidden error
    hiddenError[h] = 0.0;
    for (unsigned o = 0; o < noOutputs; o++) {
      hiddenError[h] += outputError[o] * hiddenOutput[h][o] * hidden[h] * (1.0 - hidden[h]);
    }

    // correct input - hidden
    for (unsigned i = 0; i < noInputs; i++) {
      inputHidden_dw[i][h] += eta * hiddenError[h] * input[i];
    }
    //bias - frans didn't have this
    inputHidden_dw[noInputs][h] += eta * hiddenError[h];
  }

  delete [] outputError;
  delete [] hiddenError;
  return sse;
}

void NeuralNetwork::updateWeights() {
  for (unsigned i = 0; i < noInputs + 1; i++)
    for (unsigned h = 0; h < noHiddens; h++) {
      inputHidden[i][h] += alpha * inputHidden_dwprev[i][h] + inputHidden_dw[i][h];
      inputHidden_dwprev[i][h] = inputHidden_dw[i][h];
    }
  for (unsigned h = 0; h < noHiddens + 1; h++)
    for (unsigned o = 0; o < noOutputs; o++) {
      hiddenOutput[h][o] += alpha * hiddenOutput_dwprev[h][o] + hiddenOutput_dw[h][o];
      hiddenOutput_dwprev[h][o] = hiddenOutput_dw[h][o];
    }
}

double NeuralNetwork::calculateSSE(const double *input, 
				   const double *hidden, 
				   const double *output) {

  double *outputError = new double[noOutputs];
  double sse = 0.0;
  for (unsigned o = 0; o < noOutputs; o++) {
    outputError[o] = input[noInputs + o] - output[o];
    sse += outputError[o] * outputError[o];
  }
  sse = sse / (double) noOutputs;
  delete [] outputError;
  return sse;
}

void NeuralNetwork::pickPatterns() {
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

void NeuralNetwork::trainingPatterns() {
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

void NeuralNetwork::validationPatterns() {
  if (sample != NULL)
    delete sample;
  sample = new DynamicArray<Pattern *>(*(data->GetValidationSet()));
}

void NeuralNetwork::generalizationPatterns() {
  if (sample != NULL)
    delete sample;
  sample = new DynamicArray<Pattern *>(*(data->GetGeneralizationSet()));
}

void NeuralNetwork::shuffle() {
  for (unsigned i = 0; i < sample->Length(); i++) {
    unsigned a = Random::GetRandom(0U, sample->Length());
    unsigned b = Random::GetRandom(0U, sample->Length());
    Pattern * tmp = (*sample)[a];
    (*sample)[a] = (*sample)[b];
    (*sample)[b] = tmp;
  }
}

void NeuralNetwork::Optimize(double * outArray, char *fn) {
  double *input = new double[noInputs + noOutputs];
  double *hidden = new double[noHiddens];
  double *output = new double[noOutputs];
  double mse = 0;
  char *mask = data->GetAttributeMask();

  //trainingPatterns();
  for (unsigned n = 0; n < epochs; n++) {
    mse = 0;
    pickPatterns();
    shuffle();
    for (unsigned p = 0; p < sample->Length(); p++) {
      clearDeltas();

      formatPattern(*((*sample)[p]), mask, input);
      for (unsigned h = 0; h < noHiddens; h++)
	hidden[h] = 0.0;
      for (unsigned o = 0; o < noOutputs; o++)
	output[o] = 0.0;

      forwardPropagate(input, hidden, output);
      mse += backPropagate(input, hidden, output);

      updateWeights();
    }
    mse /= sample->Length();
#if SHOW_NEURALNETWORK == 1
    cout << n << " MSE: " << mse << endl;
#endif
  }

  ofstream fout(fn);
  mse = 0;
  trainingPatterns();
  for (unsigned p = 0; p < sample->Length(); p++) {
    formatPattern(*((*sample)[p]), mask, input);
    for (unsigned h = 0; h < noHiddens; h++)
      hidden[h] = 0.0;
    for (unsigned o = 0; o < noOutputs; o++)
      output[o] = 0.0;
    
    forwardPropagate(input, hidden, output);
    mse += calculateSSE(input, hidden, output);
    fout << *((*sample)[p]);
    fout << " " << output[0] << endl;    
  }
  mse /= sample->Length();
  outArray[0] = mse;
#if SHOW_NEURALNETWORK == 1
  cout << "TMSE: " << mse << endl;
#endif

  mse = 0;
  validationPatterns();
  for (unsigned p = 0; p < sample->Length(); p++) {
    formatPattern(*((*sample)[p]), mask, input);
    for (unsigned h = 0; h < noHiddens; h++)
      hidden[h] = 0.0;
    for (unsigned o = 0; o < noOutputs; o++)
      output[o] = 0.0;
    
    forwardPropagate(input, hidden, output);
    mse += calculateSSE(input, hidden, output);
    fout << *((*sample)[p]);
    fout << " " << output[0] << endl;    
  }
  mse /= sample->Length();
  outArray[1] = mse;
#if SHOW_NEURALNETWORK == 1
  cout << "VMSE: " << mse << endl;
#endif

  mse = 0;
  generalizationPatterns();
  for (unsigned p = 0; p < sample->Length(); p++) {
    formatPattern(*((*sample)[p]), mask, input);
    for (unsigned h = 0; h < noHiddens; h++)
      hidden[h] = 0.0;
    for (unsigned o = 0; o < noOutputs; o++)
      output[o] = 0.0;
    
    forwardPropagate(input, hidden, output);
    mse += calculateSSE(input, hidden, output);
    fout << *((*sample)[p]);
    fout << " " << output[0] << endl;    
  }
  mse /= sample->Length();
  outArray[2] = mse;
#if SHOW_NEURALNETWORK == 1
  cout << "GMSE: " << mse << endl;
#endif
  fout.close();
}
