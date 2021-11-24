#ifndef DATA_H
#define DATA_H

#include <fstream>
#include <string>
#include <ctype.h>

#include "../templates/Exceptions.h"
#include "../templates/DynamicArray.h"
#include "Pattern.h"
#include "Random.h"

class Data {
 public:
  Data(char *);
  Data(const Data &);
  ~Data();

  DynamicArray<Pattern *>* GetTrainingSet() const;
  DynamicArray<Pattern *>* GetValidationSet() const;
  DynamicArray<Pattern *>* GetGeneralizationSet() const;
  void ShufflePatterns();
  void CrossValidatePatterns();
  unsigned GetNoAttributes() const;
  char *GetAttributeMask() const;
  string& GetAttribute(unsigned) const;
  string& GetValue(unsigned) const;
  unsigned& GetColumn(unsigned) const;
  unsigned GetNoColumns() const;
  double GetMaximum(unsigned) const;
  double GetMinimum(unsigned) const;
  void TotalTraining();
  void TotalValidation();
  void TotalGeneralization();
  double GetStd() const;
  unsigned GetNoNominalAttributes();
  unsigned GetNoRealAttributes();
  unsigned GetSyntaxMode() const;
  unsigned GetCrossValidation() const;
  
#if OPT_MODE > 0
  unsigned GetNoClusters() const;
  unsigned GetNoClusterEpochs() const;
#endif

#if OPT_MODE != 4
#if OPT_MODE > 1
  double GetFuncMutationRate() const;
  double GetFuncCrossoverRate() const;
  unsigned GetNoFuncGenerations() const;
  unsigned GetNoFuncIndividuals() const;
  unsigned GetPolynomialOrder() const;
  double GetFuncPercentageSampleSize() const;
  unsigned GetFuncMaxComponents() const;
  double GetFuncElite() const;
  double GetFuncCutOff() const;
#endif

#if OPT_MODE > 2
  unsigned GetDecisionMaxNodes() const;
  double GetDecisionMEWorstVsAnyConsequent() const;
  double GetDecisionMECreateVsRedistributeLeafNodes() const;
  double GetDecisionMNAntecedentVsConsequent() const;
  double GetDecisionMNWorstVsAnyAntecedent() const;
  double GetDecisionMNWorstVsAnyConsequent() const;
  double GetDecisionReoptimizeVsSelectLeaf() const;
  double GetDecisionMutateExpand() const;
  double GetDecisionMutateShrink() const;
  double GetDecisionMutateNode() const;
  double GetDecisionMutateReinitialize() const;  
  double GetDecisionNAAttributeVsClassOptimize() const;
  double GetDecisionCAAttributeOptimize() const;
  double GetDecisionCAClassOptimize() const;
  double GetDecisionCAConditionOptimize() const;
  double GetDecisionCAClassVsGaussian() const;
  double GetDecisionCAClassPartition() const;
  double GetDecisionCAConditionalPartition() const;
  unsigned GetDecisionPoolNoClustersStart() const;
  unsigned GetDecisionPoolNoClustersDivision() const;
  unsigned GetDecisionPoolNoClusterEpochs() const;
  double GetDecisionInitialPercentageSampleSize() const;
  double GetDecisionSampleAcceleration() const;
  unsigned GetDecisionNoIndividuals() const;
  unsigned GetDecisionNoGenerations() const;
  double GetDecisionElite() const;
  double GetDecisionMutationRateInitial() const;
  double GetDecisionMutationRateIncrement() const;
  double GetDecisionMutationRateMax() const;
  double GetDecisionCrossoverRate() const;
  unsigned GetDecisionPoolFragmentLifeTime() const;
#endif
#endif

#if OPT_MODE == 4
  unsigned GetNeuralInputUnits();
  unsigned GetNeuralHiddenUnits();
  unsigned GetNeuralOutputUnits();
  double GetNeuralActivation();
  double GetNeuralLearningRate();
  double GetNeuralMomentum();
  unsigned GetNeuralEpochs();
  double GetNeuralPercentageSampleSize();
#endif

 private:
  DynamicArray<string> attributes;
  DynamicArray<char> types;
  DynamicArray<string> values;
  DynamicArray<unsigned> columns;
  DynamicArray<double> maximums;
  DynamicArray<double> minimums;
  unsigned nominalAttributes;
  unsigned realAttributes;

  DynamicArray<Pattern *> *trainingPatterns;
  DynamicArray<Pattern *> *generalPatterns;
  DynamicArray<Pattern *> *testPatterns;

  char *mask;

  string ReadString(ifstream &, char &);
  void ReadWhiteSpace(ifstream &, char &, unsigned &);
  char ReadType(ifstream &, char &, unsigned &);
  void ParseData(ifstream &);
  void ParseNominals(string, char, ifstream &, char &, unsigned &);
  void ParsePatterns(ifstream &, DynamicArray<Pattern *> *);
  string ToString(unsigned);
  unsigned syntaxMode;
  unsigned crossValidation;
  double std;

#if OPT_MODE > 0
  unsigned noClusters;
  unsigned noClusterEpochs;
#endif

#if OPT_MODE != 4
#if OPT_MODE > 1
  double funcMutationRate;
  double funcCrossoverRate;
  unsigned noFuncGenerations;
  unsigned noFuncIndividuals;
  unsigned polynomialOrder;
  double funcPercentageSampleSize;
  unsigned funcMaxComponents;
  double funcElite;
  double funcCutOff;
#endif

#if OPT_MODE > 2
  unsigned decisionMaxNodes;
  double decisionMEWorstVsAnyConsequent;
  double decisionMECreateVsRedistributeLeafNodes;
  double decisionMNAntecedentVsConsequent;
  double decisionMNWorstVsAnyAntecedent;
  double decisionMNWorstVsAnyConsequent;
  double decisionReoptimizeVsSelectLeaf;
  double decisionMutateExpand;
  double decisionMutateShrink;
  double decisionMutateNode;
  double decisionMutateReinitialize;
  double decisionNAAttributeVsClassOptimize;
  double decisionCAAttributeOptimize;
  double decisionCAClassOptimize;
  double decisionCAConditionOptimize;
  double decisionCAClassVsGaussian;
  double decisionCAClassPartition;
  double decisionCAConditionalPartition;
  unsigned decisionPoolNoClustersStart;
  unsigned decisionPoolNoClustersDivision;
  unsigned decisionPoolNoClusterEpochs;
  unsigned decisionPoolFragmentLifeTime;
  double decisionInitialPercentageSampleSize;
  double decisionSampleAcceleration;
  unsigned decisionNoIndividuals;
  unsigned decisionNoGenerations;
  double decisionElite;
  double decisionMutationRateInitial;
  double decisionMutationRateIncrement;
  double decisionMutationRateMax;
  double decisionCrossoverRate;
#endif
#endif

#if OPT_MODE == 4
  unsigned neuralInputUnits;
  unsigned neuralHiddenUnits;
  unsigned neuralOutputUnits;
  double neuralActivation;
  double neuralLearningRate;
  double neuralMomentum;
  unsigned neuralEpochs;
  double neuralPercentageSampleSize;
#endif

};

#endif

