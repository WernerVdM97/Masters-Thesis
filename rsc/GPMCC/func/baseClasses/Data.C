/* 
   Class: Data
   By:    G. Potgieter

   The data class provides the program with the parsing utilities to load
   a set of datafiles (with .names and .dat extensions)

   The EBNF for the namesfile should be something like:

   NAMES = {{WHITESPACE} ([CONTROL_VALUE] | [PATTERN_ATTRIBUTE])}
   WHITESPACE = {(' ' | '\t' | '\n')}
   CONTROL_VALUE = ([CONTROL_VARIABLE]':'[VALUE])
   CONTROL_VARIABLE = *ANY DEFINED CLASS VARIABLE*
   VALUE = *THE TYPE OF THE CONTROL VARIABLE*
   PATTERN_ATTRIBUTE = ([HEADER]':'[DATA_TYPE])
   HEADER = ('<C>'[TEXT] | [TEXT])
   DATA_TYPE = ('<continuous>' | 
               '<nominal>'[{WHITESPACE}[TEXT]]{,{WHITESPACE}[TEXT]}'.')
   TEXT = *ANY VALID ALPHA NUMERIC CHARACTER AND WHITESPACE*

   The datafile should consist of something like:
   DATA = {PATTERN}
   PATTERN = {[VALUE][WHITESPACE]}*ATTRIBUTES_TIMES*

   The bit masking meaning is slightly tricky so:

   type bit 0 -> 1 continuous / 0 nominal
        bit 1 -> 1 class / 0 attribute
*/

#include "Data.h"

Data::Data(char *filename) {
  trainingPatterns = new DynamicArray<Pattern *>();
  generalPatterns = new DynamicArray<Pattern *>();
  testPatterns = new DynamicArray<Pattern *>();
  syntaxMode = 0;
  crossValidation = 0;

#if OPT_MODE > 0
  noClusters = 0;
  noClusterEpochs = 0;
#endif

#if OPT_MODE != 4
#if OPT_MODE > 1
  funcMutationRate = 0;
  funcCrossoverRate = 0;
  noFuncGenerations = 0;
  noFuncIndividuals = 0;
  polynomialOrder = 0;
  funcPercentageSampleSize = 0;
  funcMaxComponents = 0;
  funcElite = 0;
  funcCutOff = 0;
#endif

#if OPT_MODE > 2
  decisionMaxNodes = 0;
  decisionMEWorstVsAnyConsequent = 0;
  decisionMECreateVsRedistributeLeafNodes = 0;
  decisionMNAntecedentVsConsequent = 0;
  decisionMNWorstVsAnyAntecedent = 0;
  decisionMNWorstVsAnyConsequent = 0;
  decisionReoptimizeVsSelectLeaf = 0;
  decisionMutateExpand = 0;
  decisionMutateShrink = 0;
  decisionMutateNode = 0;
  decisionMutateReinitialize = 0;
  decisionNAAttributeVsClassOptimize = 0;
  decisionCAAttributeOptimize = 0;
  decisionCAClassOptimize = 0;
  decisionCAConditionOptimize = 0;
  decisionCAClassVsGaussian = 0;
  decisionCAClassPartition = 0;
  decisionCAConditionalPartition = 0;
  decisionPoolNoClustersStart = 0;
  decisionPoolNoClustersDivision = 0;
  decisionPoolNoClusterEpochs = 0;
  decisionInitialPercentageSampleSize = 0;
  decisionSampleAcceleration = 0;
  decisionNoIndividuals = 0;
  decisionNoGenerations = 0;
  decisionElite = 0;
  decisionMutationRateInitial = 0;
  decisionMutationRateIncrement = 0;
  decisionMutationRateMax = 0;
  decisionCrossoverRate = 0;
  decisionPoolFragmentLifeTime = 0;
#endif
#endif

#if OPT_MODE == 4
  neuralInputUnits = 0;
  neuralHiddenUnits = 0;
  neuralOutputUnits = 0;
  neuralActivation = 0;
  neuralLearningRate = 0;
  neuralMomentum = 0;
  neuralEpochs = 0;
  neuralPercentageSampleSize = 0;
#endif

  // go data file
  char *file = new char[strlen(filename) + 7];
  strcpy(file, filename);
  file[strlen(filename)] = '.';
  file[strlen(filename)+1] = 'n';
  file[strlen(filename)+2] = 'a';
  file[strlen(filename)+3] = 'm';
  file[strlen(filename)+4] = 'e';
  file[strlen(filename)+5] = 's';
  file[strlen(filename)+6] = '\0';

  ifstream stream(file);
  if (stream == NULL) {
    FileException *exception = new FileException(string("invalid filename '") + file + "'");
    delete [] file;
    throw exception;
  }
  delete [] file;
  ParseData(stream);
  stream.close();

#if OPT_MODE > 0
  if (noClusters == 0)
    throw new ParseException("Number of clusters must be greater than 0");
#endif

#if OPT_MODE != 4
#if OPT_MODE > 1
  if (noFuncIndividuals == 0)
    throw new ParseException("Number of function individuals must be greater than 0");
  if ((polynomialOrder == 0) || (polynomialOrder > 255))
    throw new ParseException("Polynomial order must be between 1 and 255");
  if (funcMaxComponents == 0)
    throw new ParseException("The maximum number of function components must be greater than 0");
  if ((unsigned) (funcCrossoverRate * noFuncIndividuals) == 0)
    throw new ParseException("Function crossover rate must include at least one individual");
  if ((funcMutationRate < 0.0) || (funcMutationRate > 1.0))
    throw new ParseException("Function mutation rate must be between 0 and 1");
  if ((funcCrossoverRate < 0.0) || (funcCrossoverRate > 1.0))
    throw new ParseException("Function crossover rate must be between 0 and 1");
#endif

#if OPT_MODE > 2
  if (decisionMaxNodes == 0)
    throw new ParseException("Decision tree must have more than 0 nodes");
  if ((decisionMEWorstVsAnyConsequent < 0) || (decisionMEWorstVsAnyConsequent > 1))
    throw new ParseException("Decision ME worst vs. any consequent must be between 0 and 1");
  if ((decisionMECreateVsRedistributeLeafNodes < 0) || (decisionMECreateVsRedistributeLeafNodes > 1))
    throw new ParseException("Decision ME create vs. redistribute leaf nodes must be between 0 and 1");
  if ((decisionMNAntecedentVsConsequent < 0) || (decisionMNAntecedentVsConsequent > 1))
    throw new ParseException("Decision MN antecedent vs. consequent must be between 0 and 1");
  if ((decisionMNWorstVsAnyAntecedent < 0) || (decisionMNWorstVsAnyAntecedent > 1))
    throw new ParseException("Decision MN worst vs. any antecedent must be between 0 and 1");
  if ((decisionMNWorstVsAnyConsequent < 0) || (decisionMNWorstVsAnyConsequent > 1))
    throw new ParseException("Decision MN worst vs. any consequent must be between 0 and 1");
  if ((decisionReoptimizeVsSelectLeaf < 0) || (decisionReoptimizeVsSelectLeaf > 1))
    throw new ParseException("Decision reoptimize vs. select leaf node must be between 0 and 1");
  if ((decisionMutateExpand + decisionMutateShrink + decisionMutateNode + decisionMutateReinitialize < 0.999) || (decisionMutateExpand + decisionMutateShrink + decisionMutateNode + decisionMutateReinitialize > 1.001))
    throw new ParseException("Decision mutation components must add up to 1");
  if ((decisionNAAttributeVsClassOptimize < 0) || (decisionNAAttributeVsClassOptimize > 1))
    throw new ParseException("Decision NA attribute vs. class must be between 0 and 1");
  if ((decisionCAAttributeOptimize + decisionCAClassOptimize + decisionCAConditionOptimize < 0.999) || (decisionCAAttributeOptimize + decisionCAClassOptimize + decisionCAConditionOptimize > 1.001))
    throw new ParseException("Decision CA mutation components must add up to 1");
  if ((decisionCAClassVsGaussian < 0) || (decisionCAClassVsGaussian > 1))
    throw new ParseException("Decision CA class vs. gaussian must be between 0 and 1");
  if (decisionCAClassPartition < 0)
    throw new ParseException("Decision CA class partition must be greater than 0");
  if ((decisionCAConditionalPartition < 0) || (decisionCAConditionalPartition > 1))
    throw new ParseException("Decision CA conditional partition must be between 0 and 1");
  if (decisionPoolNoClustersStart == 0)
    throw new ParseException("Decision number of pool clusters initial must be larger than 0");
  if (decisionPoolNoClustersDivision < 2)
    throw new ParseException("Decision number of pool clusters division must be larger than 1");
  if ((decisionInitialPercentageSampleSize < 0) || (decisionInitialPercentageSampleSize > 1))
    throw new ParseException("Decision initial percentage sample size must be larger than 1");
  if (decisionNoIndividuals == 0)
    throw new ParseException("Decision number of individuals must be larger than 0");
  if (decisionNoGenerations == 0)
    throw new ParseException("Decision number of generations must be larger than 0");
  if ((decisionElite < 0) || (decisionElite > 1))
    throw new ParseException("Decision elite must be between 0 and 1");
  if ((decisionMutationRateInitial < 0) || (decisionMutationRateInitial > 1))
    throw new ParseException("Decision mutation rate initial must be between 0 and 1");
  if ((decisionMutationRateIncrement < 0) || (decisionMutationRateIncrement > 1))
    throw new ParseException("Decision mutation rate increment must be between 0 and 1");
  if ((decisionMutationRateMax < 0) || (decisionMutationRateMax > 1))
    throw new ParseException("Decision mutation rate maximum must be between 0 and 1");
  if ((decisionCrossoverRate < 0) || (decisionCrossoverRate > 1))
    throw new ParseException("Decision crossover rate maximum must be between 0 and 1");
  if (decisionPoolFragmentLifeTime == 0)
    throw new ParseException("Decision fragment life time must be greater than 0");
#endif
#endif

#if OPT_MODE == 4
  if (neuralInputUnits == 0)
    throw new ParseException("Neural network input units must be non zero");
  else if (neuralHiddenUnits == 0)
    throw new ParseException("Neural network hidden units must be non zero");
  else if (neuralOutputUnits == 0)
    throw new ParseException("Neural network output units must be non zero");
#endif

  nominalAttributes = 0;
  realAttributes = 0;
  for (unsigned i = 0; i < columns.Length(); i += 2) {
    if (types[columns[i]] == 0)
      nominalAttributes++;
    else if (types[columns[i]] == 1)
      realAttributes++;
  }

  if (nominalAttributes + realAttributes == 0)
    throw new ParseException("Not enough attributes to proceed");

  cout << "Attributes" << endl << attributes << endl;
  cout << "Types" << endl;
  for (unsigned i = 0; i < types.Length(); i++)
    cout << (unsigned) types[i] << " ";
  cout << endl;
  cout << "Values" << endl << values << endl;
  cout << "Columns" << endl << columns << endl;
  
  // go training pattern file
  file = new char[strlen(filename) + 10];
  strcpy(file, filename);
  file[strlen(filename)] = '.';
  file[strlen(filename)+1] = 't';
  file[strlen(filename)+2] = 'r';
  file[strlen(filename)+3] = 'a';
  file[strlen(filename)+4] = 'i';
  file[strlen(filename)+5] = 'n';
  file[strlen(filename)+6] = 'i';
  file[strlen(filename)+7] = 'n';
  file[strlen(filename)+8] = 'g';
  file[strlen(filename)+9] = '\0';

  ifstream stream2(file);
  if (stream2 == NULL) {
    FileException *exception = new FileException(string("invalid filename '") + file + "'");
    delete [] file;
    throw exception;
  }
  delete [] file;
  ParsePatterns(stream2, trainingPatterns);
  stream2.close();

#if OPT_MODE == 2
  if ((unsigned) (funcPercentageSampleSize * trainingPatterns->Length()) <= funcMaxComponents)
    throw new ParseException("The effective function sample size must be larger than the maximum number of function components");
#endif

  // go generalization pattern file
  file = new char[strlen(filename) + 9];
  strcpy(file, filename);
  file[strlen(filename)] = '.';
  file[strlen(filename)+1] = 'g';
  file[strlen(filename)+2] = 'e';
  file[strlen(filename)+3] = 'n';
  file[strlen(filename)+4] = 'e';
  file[strlen(filename)+5] = 'r';
  file[strlen(filename)+6] = 'a';
  file[strlen(filename)+7] = 'l';
  file[strlen(filename)+8] = '\0';

  ifstream stream3(file);
  if (stream3 == NULL) {
    FileException *exception = new FileException(string("invalid filename '") + file + "'");
    delete [] file;
    throw exception;
  }
  delete [] file;
  ParsePatterns(stream3, generalPatterns);
  stream3.close();

  // go test pattern file
  file = new char[strlen(filename) + 6];
  strcpy(file, filename);
  file[strlen(filename)] = '.';
  file[strlen(filename)+1] = 't';
  file[strlen(filename)+2] = 'e';
  file[strlen(filename)+3] = 's';
  file[strlen(filename)+4] = 't';
  file[strlen(filename)+5] = '\0';

  ifstream stream4(file);
  if (stream4 == NULL) {
    FileException *exception = new FileException(string("invalid filename '") + file + "'");
    delete [] file;
    throw exception;
  }
  delete [] file;
  ParsePatterns(stream4, testPatterns);
  stream4.close();

  cout << "Minimums" << endl << minimums << endl;
  cout << "Maximums" << endl << maximums << endl;
  cout << "Number of training patterns: " << trainingPatterns->Length() << endl;
  cout << "Number of validation patterns: " << generalPatterns->Length() << endl;
  cout << "Number of generalization patterns: " << testPatterns->Length() << endl;
}

Data::~Data() {
  for (unsigned i = 0; i < trainingPatterns->Length(); i++)
    delete (*trainingPatterns)[i];
  delete trainingPatterns;
  for (unsigned i = 0; i < generalPatterns->Length(); i++)
    delete (*generalPatterns)[i];
  delete generalPatterns;
  for (unsigned i = 0; i < testPatterns->Length(); i++)
    delete (*testPatterns)[i];
  delete testPatterns;
  delete [] mask;
}

string Data::ReadString(ifstream &stream, char &c) {
  string str;
  while ((c != ':') && (!stream.eof())) {
    str += c;
    c = stream.get();
  }
  return str;
}

void Data::ReadWhiteSpace(ifstream &stream, char &c, unsigned &line) {
  while ((isspace(c)) && (!stream.eof())) {
    if (c == '\n')
      line++;
    c = stream.get();
  }
}

char Data::ReadType(ifstream &stream, char &c, unsigned &line) {
  string str;
  if ((c != '<') || (stream.eof()))
    throw new ParseException("type identifier expected on line " + ToString(line));
  c = stream.get();
  while ((isalnum(c)) && (!stream.eof())) {
    str += c;
    c = stream.get();
  }
  if ((c != '>') || (stream.eof()))
    throw new ParseException("type identifier not terminated on line " + ToString(line));
  c = stream.get();

  if (str.find("continuous") == 0)
    return 1;
  else if (str.find("nominal") == 0)
    return 0;
  else
    throw new ParseException("invalid type identifier on line " + ToString(line));
  return 0;
}

void Data::ParseNominals(string str, char type, ifstream &stream, char &c, unsigned &line) {
  ReadWhiteSpace(stream, c, line);
  if ((c == ',') || (c == '.'))
    throw new ParseException("Nominal type expected on line "  + ToString(line));
  do {
    ReadWhiteSpace(stream, c, line);
    string str2 = "";
    while ((isalnum(c)) && (!stream.eof())) {
      str2 += c;
      c = stream.get();
    }
    if (stream.eof())
      throw new ParseException("',' or '.' expected on line " + ToString(line));
    if ((c != ',') && (c != '.'))
      throw new ParseException("',' or '.' expected");
    attributes.Add(str);
    minimums.Add(1.0e200);
    maximums.Add(-1.0e200);
    types.Add(type);
    values.Add(str2);
    if (c == ',') {
      c = stream.get();
      if (stream.eof())
	throw new ParseException("Nominal type expected on line " + ToString(line));
    }
  } while ((c != '.') && (!stream.eof()));
  if (c != '.')
    throw new ParseException("'.' expected on line "  + ToString(line));
  columns.Add(attributes.Length());
  c = stream.get();
}


DynamicArray<Pattern *>* Data::GetTrainingSet() const {
  return trainingPatterns;
}

DynamicArray<Pattern *>* Data::GetValidationSet() const {
  return generalPatterns;
}

DynamicArray<Pattern *>* Data::GetGeneralizationSet() const {
  return testPatterns;
}

void Data::ShufflePatterns() {
  for (unsigned i = 0; i < 3 * trainingPatterns->Length(); i++) {
    unsigned a = Random().GetRandom(0, trainingPatterns->Length());
    unsigned b = Random().GetRandom(0, trainingPatterns->Length());
    Pattern *tmp = (*trainingPatterns)[a];
    (*trainingPatterns)[a] = (*trainingPatterns)[b];
    (*trainingPatterns)[b] = tmp;
  }
  for (unsigned i = 0; i < 3 * testPatterns->Length(); i++) {
    unsigned a = Random().GetRandom(0, testPatterns->Length());
    unsigned b = Random().GetRandom(0, testPatterns->Length());
    Pattern *tmp = (*testPatterns)[a];
    (*testPatterns)[a] = (*testPatterns)[b];
    (*testPatterns)[b] = tmp;
  }
}

void Data::CrossValidatePatterns() {
  unsigned trainingLength = trainingPatterns->Length();
  unsigned setLength = generalPatterns->Length();
  unsigned upperLength = trainingLength - setLength;
  Pattern **tmp;

  tmp = new Pattern *[setLength];
  for (unsigned i = 0; i < setLength; i++)
    tmp[i] = (*trainingPatterns)[i];
  for (unsigned i = 0; i < upperLength; i++)
    (*trainingPatterns)[i] = (*trainingPatterns)[i + setLength];
  for (unsigned i = 0; i < setLength; i++)
    (*trainingPatterns)[i + upperLength] = (*generalPatterns)[i];
  for (unsigned i = 0; i < setLength; i++)
    (*generalPatterns)[i] = tmp[i];
  delete [] tmp;

  setLength = testPatterns->Length();
  upperLength = trainingLength - setLength;
  tmp = new Pattern *[setLength];
  for (unsigned i = 0; i < setLength; i++)
    tmp[i] = (*trainingPatterns)[i];
  for (unsigned i = 0; i < upperLength; i++)
    (*trainingPatterns)[i] = (*trainingPatterns)[i + setLength];
  for (unsigned i = 0; i < setLength; i++)
    (*trainingPatterns)[i + upperLength] = (*testPatterns)[i];
  for (unsigned i = 0; i < setLength; i++)
    (*testPatterns)[i] = tmp[i];
  delete [] tmp;
  
  /*
  setLength = trainingPatterns->Length();
  for (unsigned i = 0; i < setLength; i++)
    cout << (*trainingPatterns)[i] << endl;
  setLength = generalPatterns->Length();
  for (unsigned i = 0; i < setLength; i++)
    cout << (*generalPatterns)[i] << endl;
  setLength = testPatterns->Length();
  for (unsigned i = 0; i < setLength; i++)
    cout << (*testPatterns)[i] << endl;
  */
}

void Data::ParseData(ifstream &stream) {
  char c = ' ';
  unsigned line = 0;
  while (!stream.eof()) {
    ReadWhiteSpace(stream, c, line);
    if (stream.eof())
      break;
    string str = ReadString(stream, c);
    if ((c != ':') || (stream.eof()))
      throw new ParseException("':' expected on line " + ToString(line));

    if (str.find("SyntaxMode") == 0) {
      stream >> syntaxMode;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("CrossValidation") == 0) {
      stream >> crossValidation;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else
#if OPT_MODE > 0
    if (str.find("Clusters") == 0) {
      stream >> noClusters;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("ClusterEpochs") == 0) {
      stream >> noClusterEpochs;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else
#endif
#if OPT_MODE != 4
#if OPT_MODE > 1
    if (str.find("FunctionMutationRate") == 0) {
      stream >> funcMutationRate;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("FunctionCrossoverRate") == 0) {
      stream >> funcCrossoverRate;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("FunctionGenerations") == 0) {
      stream >> noFuncGenerations;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("FunctionIndividuals") == 0) {
      stream >> noFuncIndividuals;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("PolynomialOrder") == 0) {
      stream >> polynomialOrder;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("FunctionPercentageSampleSize") == 0) {
      stream >> funcPercentageSampleSize;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("FunctionMaximumComponents") == 0) {
      stream >> funcMaxComponents;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("FunctionElite") == 0) {
      stream >> funcElite;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("FunctionCutOff") == 0) {
      stream >> funcCutOff;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else
#endif
#if OPT_MODE > 2
    if (str.find("DecisionMaxNodes") == 0) {
      stream >> decisionMaxNodes;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionMEWorstVsAnyConsequent") == 0) {
      stream >> decisionMEWorstVsAnyConsequent;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionMECreateVsRedistributeLeafNodes") == 0) {
      stream >> decisionMECreateVsRedistributeLeafNodes;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionMNAntecedentVsConsequent") == 0) {
      stream >> decisionMNAntecedentVsConsequent;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionMNWorstVsAnyAntecedent") == 0) {
      stream >> decisionMNWorstVsAnyAntecedent;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionMNWorstVsAnyConsequent") == 0) {
      stream >> decisionMNWorstVsAnyConsequent;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionReoptimizeVsSelectLeaf") == 0) {
      stream >> decisionReoptimizeVsSelectLeaf;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionMutateExpand") == 0) {
      stream >> decisionMutateExpand;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionMutateShrink") == 0) {
      stream >> decisionMutateShrink;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionMutateNode") == 0) {
      stream >> decisionMutateNode;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionMutateReinitialize") == 0) {
      stream >> decisionMutateReinitialize;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionNAAttributeVsClassOptimize") == 0) {
      stream >> decisionNAAttributeVsClassOptimize;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionCAAttributeOptimize") == 0) {
      stream >> decisionCAAttributeOptimize;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionCAClassOptimize") == 0) {
      stream >> decisionCAClassOptimize;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionCAConditionOptimize") == 0) {
      stream >> decisionCAConditionOptimize;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionCAClassVsGaussian") == 0) {
      stream >> decisionCAClassVsGaussian;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionCAClassPartition") == 0) {
      stream >> decisionCAClassPartition;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionCAConditionalPartition") == 0) {
      stream >> decisionCAConditionalPartition;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionPoolNoClustersStart") == 0) {
      stream >> decisionPoolNoClustersStart;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionPoolNoClustersDivision") == 0) {
      stream >> decisionPoolNoClustersDivision;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionPoolNoClusterEpochs") == 0) {
      stream >> decisionPoolNoClusterEpochs;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionInitialPercentageSampleSize") == 0) {
      stream >> decisionInitialPercentageSampleSize;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionSampleAcceleration") == 0) {
      stream >> decisionSampleAcceleration;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionNoIndividuals") == 0) {
      stream >> decisionNoIndividuals;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionNoGenerations") == 0) {
      stream >> decisionNoGenerations;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionElite") == 0) {
      stream >> decisionElite;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionMutationRateInitial") == 0) {
      stream >> decisionMutationRateInitial;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionMutationRateIncrement") == 0) {
      stream >> decisionMutationRateIncrement;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionMutationRateMax") == 0) {
      stream >> decisionMutationRateMax;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionCrossoverRate") == 0) {
      stream >> decisionCrossoverRate;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("DecisionPoolFragmentLifeTime") == 0) {
      stream >> decisionPoolFragmentLifeTime;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else
#endif
#endif
#if OPT_MODE == 4
    if (str.find("NeuralInputUnits") == 0) {
      stream >> neuralInputUnits;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("NeuralHiddenUnits") == 0) {
      stream >> neuralHiddenUnits;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("NeuralOutputUnits") == 0) {
      stream >> neuralOutputUnits;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("NeuralActivation") == 0) {
      stream >> neuralActivation;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("NeuralLearningRate") == 0) {
      stream >> neuralLearningRate;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("NeuralMomentum") == 0) {
      stream >> neuralMomentum;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("NeuralEpochs") == 0) {
      stream >> neuralEpochs;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else if (str.find("NeuralPercentageSampleSize") == 0) {
      stream >> neuralPercentageSampleSize;
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
    } else
#endif
    if (str.find("<C>") == 0) {
      c = stream.get();
      if (stream.eof())
	throw new ParseException("value expected on line " + ToString(line));
      char x = ReadType(stream, c, line);
      columns.Add(attributes.Length());
      if (x == 1) {
	attributes.Add(str);
	minimums.Add(1.0e200);
	maximums.Add(-1.0e200);
	types.Add(x | 2);
	values.Add("");
	columns.Add(attributes.Length());
      } else {
	ParseNominals(str, x | 2, stream, c, line);
      }
    } else {
      c = stream.get();
      if (stream.eof())
	throw new ParseException("type expected on line " + ToString(line));
      char x = ReadType(stream, c, line);
      columns.Add(attributes.Length());
      if (x == 1) {
	attributes.Add(str);
	minimums.Add(1.0e200);
	maximums.Add(-1.0e200);
	types.Add(x);
	values.Add("");
	columns.Add(attributes.Length());
      } else {
	ParseNominals(str, x, stream, c, line);
      }
    }
  }
}


void Data::ParsePatterns(ifstream &stream, DynamicArray<Pattern *> *patterns) {
  mask = new char[types.Length()];
  for (unsigned i = 0; i < types.Length(); i++) {
    mask[i] = types[i];
  }

  unsigned j = 0;
  while (true) {
    double *db = new double[attributes.Length()];
    for (unsigned i = 0; i < (columns.Length() >> 1); i++) {
      unsigned find = columns[i << 1];
      if ((types[find] & 1) == 1) {
	char c;
	while (true) {
	  c = stream.peek();
	  if (stream.eof()) {
	    delete [] db;
	    return;
	  }
	  if (!isspace(c)) {
	    break;
	  }
	  stream.get();
	}
	bool found = false;
	found = (stream >> db[find]);
	if ((!found) && (!stream.eof())) {
	  delete [] db;
	  throw new ParseException(string("Unknown continuous attribute value on line ") + ToString(j));
	} else if ((!found) && (stream.eof())) {
	  delete [] db;
	  return;
	}
      } else {
	string str = "";
	char c = ' ';
	while (true) {
	  c = stream.peek();
	  if (stream.eof()) {
	    delete [] db;
	    return;
	  }
	  if (!isspace(c)) {
	    break;
	  }
	  stream.get();
	}

	while (true) {
	  c = stream.peek();
	  if (stream.eof()) {
	    break;
	  }
	  if (!isalnum(c)) {
	    break;
	  }
	  str += c;
	  stream.get();
	}
	stream.get();

	bool found = false;
	for (unsigned k = columns[i << 1]; k < columns[(i << 1) + 1]; k++) {
	  if (str.find(values[k]) == 0) {
	    db[k] = 1.0;
	    found = true;
	  } else
	    db[k] = 0.0;
	}
	if ((!found) && (!stream.eof())) {
	  delete [] db;
	  throw new ParseException(string("Unknown nominal attribute value '") + str + "' on line " + ToString(j));
	} else if ((!found) && (stream.eof())) {
	  delete [] db;
	  return;
	}
      }
    }

    for (unsigned i = 0; i < attributes.Length(); i++) {
      if (db[i] < minimums[i])
	minimums[i] = db[i];
      if (db[i] > maximums[i])
	maximums[i] = db[i];
    }
    patterns->Add(new Pattern(db, attributes.Length(), mask));
    j++;
    delete [] db;
  }
}

unsigned Data::GetNoAttributes() const {
  return attributes.Length();
}

char * Data::GetAttributeMask() const {
  return mask;
}

string& Data::GetAttribute(unsigned attribute) const {
  if (attribute >= attributes.Length())
    throw new IndexOutOfBoundsException();
  return attributes[attribute];
}

string& Data::GetValue(unsigned attribute) const {
  if (attribute >= values.Length())
    throw new IndexOutOfBoundsException();
  return values[attribute];
}

unsigned& Data::GetColumn(unsigned column) const {
  if (column >= columns.Length())
    throw new IndexOutOfBoundsException();
  return columns[column];
}

unsigned Data::GetNoColumns() const {
  return columns.Length();
}

double Data::GetMinimum(unsigned i) const {
  if (i >= minimums.Length())
    throw new IndexOutOfBoundsException();
  return minimums[i];
}

double Data::GetMaximum(unsigned i) const {
  if (i >= maximums.Length())
    throw new IndexOutOfBoundsException();
  return maximums[i];
}

void Data::TotalTraining() {
  unsigned classification = 0;
  unsigned length = attributes.Length();
  for (unsigned i = 0; i < length; i++)
    if (mask[i] == 3) {
      classification = i;
      break;
    }
  length = trainingPatterns->Length();
  double tmp;
  double sum = 0;
  double sumsq = 0;
  double minimum = minimums[classification];
  double range = maximums[classification] - minimum;
  for (unsigned i = 0; i < length; i++) {
    tmp = (*(*trainingPatterns)[i])[classification];
    sum += (tmp - minimum);
    sumsq += (tmp - minimum)*(tmp - minimum);
  }
  std = (sumsq - sum*sum / (double) length) / (range * range);
}

void Data::TotalValidation() {
  unsigned classification = 0;
  unsigned length = attributes.Length();
  for (unsigned i = 0; i < length; i++)
    if (mask[i] == 3) {
      classification = i;
      break;
    }
  length = generalPatterns->Length();
  double tmp;
  double sum = 0;
  double sumsq = 0;
  double minimum = minimums[classification];
  double range = maximums[classification] - minimum;
  for (unsigned i = 0; i < length; i++) {
    tmp = (*(*generalPatterns)[i])[classification];
    sum += (tmp - minimum);
    sumsq += (tmp - minimum)*(tmp - minimum);
  }
  std = (sumsq - sum*sum / (double) length) / (range * range);
}

void Data::TotalGeneralization() {
  unsigned classification = 0;
  unsigned length = attributes.Length();
  for (unsigned i = 0; i < length; i++)
    if (mask[i] == 3) {
      classification = i;
      break;
    }
  length = testPatterns->Length();
  double tmp;
  double sum = 0;
  double sumsq = 0;
  double minimum = minimums[classification];
  double range = maximums[classification] - minimum;
  for (unsigned i = 0; i < length; i++) {
    tmp = (*(*testPatterns)[i])[classification];
    sum += (tmp - minimum);
    sumsq += (tmp - minimum)*(tmp - minimum);
  }
  std = (sumsq - sum*sum / (double) length) / (range * range);
}

double Data::GetStd() const {
  return std;
}

unsigned Data::GetNoNominalAttributes() {
  return nominalAttributes;
}

unsigned Data::GetNoRealAttributes() {
  return realAttributes;
}

unsigned Data::GetSyntaxMode() const {
  return syntaxMode;
}

unsigned Data::GetCrossValidation() const {
  return crossValidation;
}

#if OPT_MODE > 0
unsigned Data::GetNoClusters() const {
  return noClusters;
}

unsigned Data::GetNoClusterEpochs() const {
  return noClusterEpochs;
}
#endif

#if OPT_MODE != 4
#if OPT_MODE > 1
double Data::GetFuncMutationRate() const {
  return funcMutationRate;
}

double Data::GetFuncCrossoverRate() const {
  return funcCrossoverRate;
}
  
unsigned Data::GetNoFuncGenerations() const {
  return noFuncGenerations;
}

unsigned Data::GetNoFuncIndividuals() const {
  return noFuncIndividuals;
}

unsigned Data::GetPolynomialOrder() const {
  return polynomialOrder;
}

double Data::GetFuncPercentageSampleSize() const {
  return funcPercentageSampleSize;
}

unsigned Data::GetFuncMaxComponents() const {
  return funcMaxComponents;
}

double Data::GetFuncElite() const {
  return funcElite;
}

double Data::GetFuncCutOff() const {
  return funcCutOff;
}
#endif

#if OPT_MODE > 2
unsigned Data::GetDecisionMaxNodes() const {
  return decisionMaxNodes;
}

double Data::GetDecisionMEWorstVsAnyConsequent() const {
  return decisionMEWorstVsAnyConsequent;
}

double Data::GetDecisionMECreateVsRedistributeLeafNodes() const {
  return decisionMECreateVsRedistributeLeafNodes;
}

double Data::GetDecisionMNAntecedentVsConsequent() const {
  return decisionMNAntecedentVsConsequent;
}

double Data::GetDecisionMNWorstVsAnyAntecedent() const {
  return decisionMNWorstVsAnyAntecedent;
}

double Data::GetDecisionMNWorstVsAnyConsequent() const {
  return decisionMNWorstVsAnyConsequent;
}

double Data::GetDecisionReoptimizeVsSelectLeaf() const {
  return decisionReoptimizeVsSelectLeaf;
}

double Data::GetDecisionMutateExpand() const {
  return decisionMutateExpand;
}

double Data::GetDecisionMutateShrink() const {
  return decisionMutateShrink;
}

double Data::GetDecisionMutateNode() const {
  return decisionMutateNode;
}

double Data::GetDecisionMutateReinitialize() const {
  return decisionMutateReinitialize;
}

double Data::GetDecisionNAAttributeVsClassOptimize() const {
  return decisionNAAttributeVsClassOptimize;
}

double Data::GetDecisionCAAttributeOptimize() const {
  return decisionCAAttributeOptimize;
}

double Data::GetDecisionCAClassOptimize() const {
  return decisionCAClassOptimize;
}

double Data::GetDecisionCAConditionOptimize() const {
  return decisionCAConditionOptimize;
}

double Data::GetDecisionCAClassVsGaussian() const {
  return decisionCAClassVsGaussian;
}

double Data::GetDecisionCAClassPartition() const {
  return decisionCAClassPartition;
}

double Data::GetDecisionCAConditionalPartition() const {
  return decisionCAConditionalPartition;
}

unsigned Data::GetDecisionPoolNoClustersStart() const {
  return decisionPoolNoClustersStart;
}

unsigned Data::GetDecisionPoolNoClustersDivision() const {
  return decisionPoolNoClustersDivision;
}

unsigned Data::GetDecisionPoolNoClusterEpochs() const {
  return decisionPoolNoClusterEpochs;
}

double Data::GetDecisionInitialPercentageSampleSize() const {
  return decisionInitialPercentageSampleSize;
}

double Data::GetDecisionSampleAcceleration() const {
  return decisionSampleAcceleration;
}

unsigned Data::GetDecisionNoIndividuals() const {
  return decisionNoIndividuals;
}

unsigned Data::GetDecisionNoGenerations() const {
  return decisionNoGenerations;
}

double Data::GetDecisionElite() const {
  return decisionElite;
}

double Data::GetDecisionMutationRateInitial() const {
  return decisionMutationRateInitial;
}

double Data::GetDecisionMutationRateIncrement() const {
  return decisionMutationRateIncrement;
}

double Data::GetDecisionMutationRateMax() const {
  return decisionMutationRateMax;
}

double Data::GetDecisionCrossoverRate() const {
  return decisionCrossoverRate;
}

unsigned Data::GetDecisionPoolFragmentLifeTime() const {
  return decisionPoolFragmentLifeTime;
}
#endif
#endif

#if OPT_MODE == 4
unsigned Data::GetNeuralInputUnits() {
  return neuralInputUnits;
}

unsigned Data::GetNeuralHiddenUnits() {
  return neuralHiddenUnits;
}

unsigned Data::GetNeuralOutputUnits() {
  return neuralOutputUnits;
}

double Data::GetNeuralActivation() {
  return neuralActivation;
}

double Data::GetNeuralLearningRate() {
  return neuralLearningRate;
}

double Data::GetNeuralMomentum() {
  return neuralMomentum;
}

unsigned Data::GetNeuralEpochs() {
  return neuralEpochs;
}

double Data::GetNeuralPercentageSampleSize() {
  return neuralPercentageSampleSize;
}
#endif

string Data::ToString(unsigned number) {
  string str = "";
  do {
    str = string("") + (char) (number % 10 + 48) + str;
    number /= 10;
  } while (number != 0);
  return str;
}
