#include "baseClasses/Data.h"
#include "gpdecision/GPDecision.h"
#include <fstream>

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

double tt = 0;
double stt = 0;
double sf = 0;
double ssqf = 0;
double st = 0;
double ssqt = 0;
double sv = 0;
double ssqv = 0;
double sg = 0;
double ssqg = 0;
double sn = 0;
double ssqn = 0;
double stcd = 0;
double ssqtcd = 0;
double stacd = 0;
double ssqtacd = 0;
double svcd = 0;
double ssqvcd = 0;
double svacd = 0;
double ssqvacd = 0;
double sgcd = 0;
double ssqgcd = 0;
double sgacd = 0;
double ssqgacd = 0;
double stmae = 0;
double svmae = 0;
double sgmae = 0;
double ssqtmae = 0;
double ssqvmae = 0;
double ssqgmae = 0;
double sterms = 0;
double ssqterms = 0;
double srules = 0;
double ssqrules = 0;
double sconditions = 0;
double ssqconditions = 0;

void outputSimulation(unsigned n) {
  ofstream file(simfilename);
  file << n << " ";
  file << sf << " ";
  file << ssqf << " ";
  file << st << " ";
  file << ssqt << " ";
  file << sv << " ";
  file << ssqv << " ";
  file << sg << " ";
  file << ssqg << " ";
  file << sn << " ";
  file << ssqn << " ";
  file << stcd << " ";
  file << ssqtcd << " ";
  file << svcd << " ";
  file << ssqvcd << " ";
  file << sgcd << " ";
  file << ssqgcd << " ";
  file << stacd << " ";
  file << ssqtacd << " ";
  file << svacd << " ";
  file << ssqvacd << " ";
  file << sgacd << " ";
  file << ssqgacd << " ";
  file << tt << " ";
  file << stt << " ";
  file << stmae << " ";
  file << svmae << " ";
  file << sgmae << " ";
  file << ssqtmae << " ";
  file << ssqvmae << " ";
  file << ssqgmae << " ";
  file << sterms << " ";
  file << ssqterms << " ";
  file << srules << " ";
  file << ssqrules << " ";
  file << sconditions << " ";
  file << ssqconditions << " ";

  file.close();
}

unsigned inputSimulation() {
  ifstream file(simfilename);
  if (file == NULL)
    return 0;
  unsigned n;
  file >> n;
  file >> sf;
  file >> ssqf;
  file >> st;
  file >> ssqt;
  file >> sv;
  file >> ssqv;
  file >> sg;
  file >> ssqg;
  file >> sn;
  file >> ssqn;
  file >> stcd;
  file >> ssqtcd;
  file >> svcd;
  file >> ssqvcd;
  file >> sgcd;
  file >> ssqgcd;
  file >> stacd;
  file >> ssqtacd;
  file >> svacd;
  file >> ssqvacd;
  file >> sgacd;
  file >> ssqgacd;
  file >> tt;
  file >> stt;
  file >> stmae;
  file >> svmae;
  file >> sgmae;
  file >> ssqtmae;
  file >> ssqvmae;
  file >> ssqgmae;
  file >> sterms;
  file >> ssqterms;
  file >> srules;
  file >> ssqrules;
  file >> sconditions;
  file >> ssqconditions;
  file.close();
  return n + 1;
}

void outputBest(DecisionTree *best) {
  ofstream file(bestfilename, ios::app);
  file << *best;
  file << "FIT: " << best->GetFitness() << endl;
  file << "NODES: " << best->GetNoNodes() << endl;
  double t, r, c;
  best->CalculateMiscellaneous(t, r, c);
  file << "TERMS: " << t << endl;
  file << "RULES: " << r << endl;
  file << "CONDITIONS: " << c << endl;
  file << "TMSE: " << best->GetTMSE();
  file << " VMSE: " << best->GetVMSE();
  file << " GMSE: " << best->GetGMSE() << endl;
  file << "TACD: " << best->GetTACD();
  file << " VACD: " << best->GetVACD();
  file << " GACD: " << best->GetGACD() << endl;
  file << "TCD: " << best->GetTCD();
  file << " VCD: " << best->GetVCD();
  file << " GCD: " << best->GetGCD() << endl;
  file << "TMAE: " << best->GetTMAE();
  file << " VMAE: " << best->GetVMAE();
  file << " GMAE: " << best->GetGMAE() << endl;
  file.close();
}

void outputResults(unsigned max, ostream &file) {
  file << "FIT: " << sf / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqf - (sf*sf) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "NODES: " << sn / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqn - (sn*sn) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "TERMS: " << sterms / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqterms - (sterms*sterms) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "RULES: " << srules / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqrules - (srules*srules) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "CONDITIONS: " << sconditions / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqconditions - (sconditions*sconditions) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "TMSE: " << st / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqt - (st*st) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "VMSE: " << sv / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqv - (sv*sv) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "GMSE: " << sg / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqg - (sg*sg) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "TCD: " << stcd / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqtcd - (stcd*stcd) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "VCD: " << svcd / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqvcd - (svcd*svcd) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "GCD: " << sgcd / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqgcd - (sgcd*sgcd) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "TACD: " << stacd / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqtacd - (stacd*stacd) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "VACD: " << svacd / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqvacd - (svacd*svacd) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "GACD: " << sgacd / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqgacd - (sgacd*sgacd) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "TMAE: " << stmae / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqtmae - (stmae*stmae) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "VMAE: " << svmae / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqvmae - (svmae*svmae) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "GMAE: " << sgmae / (double) max;
  if (max > 1)
    file << " " << sqrt((ssqgmae - (sgmae*sgmae) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

  file << "TIME: " << tt / (double) max;
  if (max > 1)
    file << " " << sqrt((stt - (tt*tt) / (double) max) / (max - 1.0)) << endl;
  else
    file << " 0" << endl;

}

int main(int argc, char** argv) {
  try {
    if (argc == 1) {
      cout << "Usage: " << argv[0] << " <Data_name> [simulations]" << endl;
      return 0;
    }
    unsigned length = 1;
    if (argc == 3)
      length = atoi(argv[2]);
    Random::Seed();

    setFileNames(argv[1]);
    //cout << simfilename << " " << bestfilename << " " << resultfilename << endl;

    Data *data = new Data(argv[1]);
    unsigned n = (length > 1)?inputSimulation():0;
    if (data->GetCrossValidation())
      for (unsigned i = 0; i < n; i++)
	data->CrossValidatePatterns();
    for (; n < length; n++) {
      if (length > 1)
	cout << "Simulation: " << (n + 1) << " of " << length << endl;
      clock_t start = clock();
      time_t startBig = time(NULL);
      GPDecision decision(data);
      DecisionTree *best = decision.Optimize();
      clock_t end = clock();
      time_t endBig = time(NULL);
      if (endBig - startBig > 100)
	cout << "Time: " << ((double) (endBig - startBig)) << "s" << endl;
      else
	cout << "Time: " << ((double) (end - start)) / CLOCKS_PER_SEC << "s" << endl;
      if (length > 1) {
	sf += best->GetFitness();
	ssqf += best->GetFitness()*best->GetFitness();

	st += best->GetTMSE();
	ssqt += best->GetTMSE()*best->GetTMSE();
	stcd += best->GetTCD();
	ssqtcd += best->GetTCD()*best->GetTCD();
	stacd += best->GetTACD();
	ssqtacd += best->GetTACD()*best->GetTACD();

	sv += best->GetVMSE();
	ssqv += best->GetVMSE()*best->GetVMSE();
	svcd += best->GetVCD();
	ssqvcd += best->GetVCD()*best->GetVCD();
	svacd += best->GetVACD();
	ssqvacd += best->GetVACD()*best->GetVACD();

	sg += best->GetGMSE();
	ssqg += best->GetGMSE()*best->GetGMSE();
	sgcd += best->GetGCD();
	ssqgcd += best->GetGCD()*best->GetGCD();
	sgacd += best->GetGACD();
	ssqgacd += best->GetGACD()*best->GetGACD();

	sn += best->GetNoNodes();
	ssqn += best->GetNoNodes()*best->GetNoNodes();

	if (endBig - startBig > 100) {
	  tt += ((double) (endBig - startBig));
	  stt += ((double) (endBig - startBig)) * ((double) (endBig - startBig));
	} else {
	  tt += ((double) (end - start)) / CLOCKS_PER_SEC;
	  stt += ((double) (end - start)) / CLOCKS_PER_SEC * ((double) (end - start)) / CLOCKS_PER_SEC;
	}
	stmae += best->GetTMAE();
	ssqtmae += best->GetTMAE()*best->GetTMAE();
	svmae += best->GetVMAE();
	ssqvmae += best->GetVMAE()*best->GetVMAE();
	sgmae += best->GetGMAE();
	ssqgmae += best->GetGMAE()*best->GetGMAE();

	double t, r, c;
	best->CalculateMiscellaneous(t, r, c);

	sterms += t;
	ssqterms += t*t;
	srules += r;
	ssqrules += r*r;
	sconditions += c;
	ssqconditions += c*c;
	
	outputSimulation(n);
	outputBest(best);
      }
      
    }
    if (length > 1) {
      ofstream file(resultfilename);
      outputResults(length, cout);
      outputResults(length, file);
      file.close();
    }
  } catch (Exception *exception) {
    cout << *exception << endl;
    delete exception;
  }
  return 0;
}
