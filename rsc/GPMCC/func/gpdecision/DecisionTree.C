#include "DecisionTree.h"

Data *DecisionTree::data = NULL;
ChromoPool *DecisionTree::chromoPool = NULL;

DecisionTree::DecisionTree(Data *data, ChromoPool *chromoPool) {
  DecisionTree::data = data;
  DecisionTree::chromoPool = chromoPool;

  tmse = -1;
  vmse = -1;
  tmae = -1;
  vmae = -1;
  tcd = 0;
  vcd = 0;
  tacd = 0;
  vacd = 0;

  // make a new node covers are not deleted by NewDecisionNode
  head = NewDecisionNode(chromoPool->GetTrainingSet(), data->GetValidationSet(), NULL);
  nodes = 3;

  // make a tree quick !
  unsigned rnd = Random::GetRandom(0U, (data->GetDecisionMaxNodes() - 3) / 2);
  for (unsigned i = 0; i < rnd; i++) {
    MutateExpand();
  }
}

DecisionTree::DecisionTree(const DecisionTree &dt) {
  head = new DecisionNode(*(dt.head));
  nodes = dt.nodes;
  tmse = dt.tmse;
  vmse = dt.vmse;
  tcd = dt.tcd;
  vcd = dt.vcd;
  tacd = dt.tacd;
  vacd = dt.vacd;
  tmae = dt.tmae;
  vmae = dt.vmae;
}

DecisionTree& DecisionTree::operator = (const DecisionTree &dt) {
  if (this == &dt)
    return *this;
  delete head;
  head = new DecisionNode(*(dt.head));
  nodes = dt.nodes;
  tmse = dt.tmse;
  vmse = dt.vmse;
  tcd = dt.tcd;
  vcd = dt.vcd;
  tacd = dt.tacd;
  vacd = dt.vacd;
  tmae = dt.tmae;
  vmae = dt.vmae;
  return *this;
}

DecisionTree::~DecisionTree() {
  delete head;
  head = NULL;
  tmse = -1;
  vmse = -1;
  tcd = 0;
  vcd = 0;
  tacd = 0;
  vacd = 0;
  tmae = -1;
  vmae = -1;
}

void DecisionTree::Print(ostream& os, unsigned depth, DecisionNode *dn) {
  for (unsigned i = 0; i < depth; i++)
    if (data->GetSyntaxMode() == 0)
      os << "\t";
    else
      os << "  ";
  if (dn->IsConsequent()) {
    dn->GetConsequent()->Print(os);
    os << endl;
  } else {
    os << "if (";
    dn->GetAntecedent()->Print(os);
    os << ") ";
    if (data->GetSyntaxMode() < 2)
      os << "{" << endl;
    else
      os << "then" << endl;
	
    Print(os, depth + 1, dn->GetAffirmative());
    for (unsigned i = 0; i < depth; i++)
    if (data->GetSyntaxMode() == 0)
      os << "\t";
    else
      os << "  ";

    if (data->GetSyntaxMode() < 2)
      os << "} ";
    os << "else ";
    if (data->GetSyntaxMode() < 2)
      os << "{";
    os << endl;
    Print(os, depth + 1, dn->GetNegative());
    if (data->GetSyntaxMode() < 2) {
      for (unsigned i = 0; i < depth; i++)
	if (data->GetSyntaxMode() == 0)
	  os << "\t";
	else
	  os << "  ";
      os << "}" << endl;
    }
  }
}

ostream& operator << (ostream& os, DecisionTree &dt) {
  if (dt.head == NULL)
    throw new Exception("head NULL in DecisionTree, <<");
  dt.Print(os, 0, dt.head);
  return os;
}

DecisionNode* DecisionTree::NewDecisionNode(DynamicArray<Pattern *> *tCover, DynamicArray<Pattern *> *vCover, Consequent *cn) {
  double nom = data->GetNoNominalAttributes();
  double real = data->GetNoRealAttributes();
  Antecedent *a;
  if (nom == 0) {
    a = new ContinuousAntecedent(data);
  } else if (real == 0) {
    a = new NominalAntecedent(data);
  } else {
    if (Random::GetRandom(0.0, 1.0) > nom / (nom + real))
      a = new NominalAntecedent(data);
    else
      a = new ContinuousAntecedent(data);
  }

  DynamicArray<Pattern *> *tCoverA = new DynamicArray<Pattern *>();
  DynamicArray<Pattern *> *tCoverN = new DynamicArray<Pattern *>();
  DynamicArray<Pattern *> *vCoverA = new DynamicArray<Pattern *>();
  DynamicArray<Pattern *> *vCoverN = new DynamicArray<Pattern *>();

  unsigned length = tCover->Length();
  for (unsigned i = 0; i < length; i++) {
    Pattern *pattern = (*tCover)[i];
    if (a->CoverPattern(*pattern))
      tCoverA->Add(pattern);
    else
      tCoverN->Add(pattern);
  }
  length = vCover->Length();
  for (unsigned i = 0; i < length; i++) {
    Pattern *pattern = (*vCover)[i];
    if (a->CoverPattern(*pattern))
      vCoverA->Add(pattern);
    else
      vCoverN->Add(pattern);
  }

  DecisionNode *tmp;
  if (Random::GetRandom(0.0, 1.0) < 0.5) {
    ContinuousConsequent *b = new ContinuousConsequent(data, chromoPool, tCoverA, vCoverA);
    Consequent *c;
    if (cn == NULL)
      c = new ContinuousConsequent(data, chromoPool, tCoverN, vCoverN);
    else {
      c = cn->Copy();
      c->SetCovers(tCoverN, vCoverN);
    }
    tmp = new DecisionNode(a, new DecisionNode(b), new DecisionNode(c));
  } else {
    ContinuousConsequent *b = new ContinuousConsequent(data, chromoPool, tCoverN, vCoverN);
    Consequent *c;
    if (cn == NULL)
      c = new ContinuousConsequent(data, chromoPool, tCoverA, vCoverA);
    else {
      c = cn->Copy();
      c->SetCovers(tCoverA, vCoverA);
    }
    tmp = new DecisionNode(a, new DecisionNode(c), new DecisionNode(b));
  }
  return tmp;
}

void DecisionTree::MutateExpand() {
  if (this->nodes + 2 > data->GetDecisionMaxNodes())
    return;

  // Perform worst expand or any expand
  bool special = (Random::GetRandom(0.0, 1.0) < data->GetDecisionMEWorstVsAnyConsequent())?(true):(false);

  DynamicArray<DecisionNode *> nodes;
  // Get the nodes lists
  if (special)
    head->ObtainWorstLeafAntecedent(nodes);
  else
    head->ObtainLeafAntecedents(nodes);

  // Set up the empty covers
  DynamicArray<Pattern *> *tCover = new DynamicArray<Pattern *>();
  DynamicArray<Pattern *> *vCover = new DynamicArray<Pattern *>();

  if (nodes.Length() > 0) {
    // if length > 0
    DecisionNode *node;

    // Get an op node
    if (special)
      node = nodes[nodes.Length() - 1];
    else
      node = nodes[Random::GetRandom(0U, nodes.Length())];

    // Get the left and right nodes
    DecisionNode *aff = node->GetAffirmative();
    DecisionNode *neg = node->GetNegative();
    
    if ((aff->IsConsequent()) && (neg->IsConsequent())) {
      //if both left and right of op nodes are antecedents
      if (special) {
	// if worst nodes mode
	double atmse = 0;
	double ntmse = 0;
	double avsse = 0;
	double nvsse = 0;
	unsigned complexity = 0;
	double atae;
	double ntae;
	double avae;
	double nvae;
	aff->CalculateFitness(atmse, avsse, complexity, atae, avae);
	neg->CalculateFitness(ntmse, nvsse, complexity, ntae, nvae);
	atmse /= aff->GetConsequent()->GetNoTrainingCover();
	ntmse /= neg->GetConsequent()->GetNoTrainingCover();
	if (atmse >= ntmse) {
	  // if left fitness is better than right
	  aff->GetCovers(tCover, vCover);
	  DecisionNode *tmp;
	  if (Random::GetRandom(0.0, 1.0) > data->GetDecisionMECreateVsRedistributeLeafNodes())
	    // move down consequent\/\/not deleted by NewDecisionNode
	    tmp = NewDecisionNode(tCover, vCover, aff->GetConsequent());
	  else
	    // make totally new dn\/\/not deleted by NewDecisionNode
	    tmp = NewDecisionNode(tCover, vCover, NULL);
	  node->SetAffirmative(tmp);
	} else {
	  // if right fitness is better than left
	  neg->GetCovers(tCover, vCover);
	  DecisionNode *tmp;
	  if (Random::GetRandom(0.0, 1.0) > data->GetDecisionMECreateVsRedistributeLeafNodes())
	    // move down consequent\/\/not deleted by NewDecisionNode
	    tmp = NewDecisionNode(tCover, vCover, neg->GetConsequent());
	  else
	    // make totally new dn\/\/not deleted by NewDecisionNode
	    tmp = NewDecisionNode(tCover, vCover, NULL);
	  node->SetNegative(tmp);
	}
      } else {
	// if any node mode
	if (Random::GetRandom(0.0, 1.0) < 0.5) {
	  // uniform right
	  aff->GetCovers(tCover, vCover);
	  DecisionNode *tmp;
	  if (Random::GetRandom(0.0, 1.0) > data->GetDecisionMECreateVsRedistributeLeafNodes())
	    // move down consequent\/\/not deleted by NewDecisionNode
	    tmp = NewDecisionNode(tCover, vCover, aff->GetConsequent());
	  else
	    // make totally new dn\/\/not deleted by NewDecisionNode
	    tmp = NewDecisionNode(tCover, vCover, NULL);
	  node->SetAffirmative(tmp);
	} else {
	  // uniform left
	  neg->GetCovers(tCover, vCover);
	  DecisionNode *tmp;
	  if (Random::GetRandom(0.0, 1.0) > data->GetDecisionMECreateVsRedistributeLeafNodes())
	    // move down consequent\/\/not deleted by NewDecisionNode
	    tmp = NewDecisionNode(tCover, vCover, neg->GetConsequent());
	  else
	    // make totally new dn\/\/not deleted by NewDecisionNode
	    tmp = NewDecisionNode(tCover, vCover, NULL);
	  node->SetNegative(tmp);
	}
      }
    } else if (aff->IsConsequent()) {
      // if only the left node is antecedent
      aff->GetCovers(tCover, vCover);
      DecisionNode *tmp;
      if (Random::GetRandom(0.0, 1.0) > data->GetDecisionMECreateVsRedistributeLeafNodes())
	    // move down consequent\/\/not deleted by NewDecisionNode
	    tmp = NewDecisionNode(tCover, vCover, aff->GetConsequent());
	  else
	    // make totally new dn\/\/not deleted by NewDecisionNode
	    tmp = NewDecisionNode(tCover, vCover, NULL);
      node->SetAffirmative(tmp);
    } else {
      // if only the right node is antecedent
      neg->GetCovers(tCover, vCover);
      DecisionNode *tmp;
      if (Random::GetRandom(0.0, 1.0) > data->GetDecisionMECreateVsRedistributeLeafNodes())
	// move down consequent\/\/not deleted by NewDecisionNode
	tmp = NewDecisionNode(tCover, vCover, neg->GetConsequent());
      else
	// make totally new dn\/\/not deleted by NewDecisionNode
	tmp = NewDecisionNode(tCover, vCover, NULL);
      node->SetNegative(tmp);
    }
    this->nodes += 2;
  } else {
    // if length <= 0
    delete head;           //\/\/not deleted by NewDecisionNode
    head = NewDecisionNode(chromoPool->GetTrainingSet(), data->GetValidationSet(), NULL);
    this->nodes = 3;
  }
  //so delete tCover + vCover now -- otherwise memory leak
  delete tCover;
  delete vCover;
}

void DecisionTree::MutateShrink() {
  DynamicArray<DecisionNode *> nodes;
  //Get the nodes list
  head->ObtainNonLeafAntecedents(nodes);

  DynamicArray<Pattern *> *tCover = new DynamicArray<Pattern *>();
  DynamicArray<Pattern *> *vCover = new DynamicArray<Pattern *>();
  // rand of nodes + 1 so we can also shrink the head
  unsigned rand = Random::GetRandom(0U, nodes.Length() + 1);
  if (rand < nodes.Length()) {
    // if rand is not the head node
    // Get the op node
    DecisionNode *node = nodes[rand];
    // Get the left and right nodes
    DecisionNode *aff = node->GetAffirmative();
    DecisionNode *neg = node->GetNegative();

    if ((!aff->IsConsequent()) && (!neg->IsConsequent())) {
      // if left and right are antecedents
      if (Random::GetRandom(0.0, 1.0) < 0.5) {
	// choose the right node uniformly
	this->nodes -= aff->GetNoNodes();
	// get the coverage back of the right node
	aff->GetCovers(tCover, vCover);
	DecisionNode *tmp;
	if (Random::GetRandom(0.0, 1.0) < 0.5) {
	  // copy right's right node - legal because right is antecedent thus guaranteed to be a node
	  tmp = new DecisionNode(*(aff->GetAffirmative())); 
	} else {
	  // copy right's left node - legal because right is antecedent thus guaranteed to be a node
	  tmp = new DecisionNode(*(aff->GetNegative())); 
	}
	// set the covers on the new nodes - SetCovers automatically deletes tCover + vCover
	tmp->SetCovers(tCover, vCover);
	// set the right node
	node->SetAffirmative(tmp);
	this->nodes += node->GetAffirmative()->GetNoNodes();
      } else {
	// choose the left node uniformly
	this->nodes -= neg->GetNoNodes();
	// get the coverage of the left node
	neg->GetCovers(tCover, vCover);
	DecisionNode *tmp;
	if (Random::GetRandom(0.0, 1.0) < 0.5) {
	  // copy left's right node - legal because right is antecedent thus guaranteed to be a node
	  tmp = new DecisionNode(*(neg->GetAffirmative())); 
	} else {
	  // copy left's left node - legal because right is antecedent thus guaranteed to be a node
	  tmp = new DecisionNode(*(neg->GetNegative())); 
	}
	// set the covers on the new nodes - SetCovers automatically deletes tCover + vCover
	tmp->SetCovers(tCover, vCover);
	// set the left node
	node->SetNegative(tmp);
	this->nodes += node->GetNegative()->GetNoNodes();
      }
    } else if (!aff->IsConsequent()) {
      // right is an antecedent
      this->nodes -= aff->GetNoNodes();
      // get right's covers back
      aff->GetCovers(tCover, vCover);
      DecisionNode *tmp;
      if (Random::GetRandom(0.0, 1.0) < 0.5) {
	// copy right's right node - legal because right is antecedent thus guaranteed to be a node
	tmp = new DecisionNode(*(aff->GetAffirmative())); 
      } else {
	// copy right's left node - legal because right is antecedent thus guaranteed to be a node
	tmp = new DecisionNode(*(aff->GetNegative())); 
      }
      // set the covers on the new nodes - SetCovers automatically deletes tCover + vCover
      tmp->SetCovers(tCover, vCover);
      // set the right node
      node->SetAffirmative(tmp);
      this->nodes += node->GetAffirmative()->GetNoNodes();
    } else {
      if (neg->IsConsequent())
	throw new Exception("Expected antecedent in DecisionTree, MutateShrink");
      // left is antecedent
      this->nodes -= neg->GetNoNodes();
      // get lefts's covers back
      neg->GetCovers(tCover, vCover);
      DecisionNode *tmp;
      if (Random::GetRandom(0.0, 1.0) < 0.5) {
	// copy lefts's right node - legal because right is antecedent thus guaranteed to be a node
	tmp = new DecisionNode(*(neg->GetAffirmative())); 
      } else {
	// copy left's left node - legal because right is antecedent thus guaranteed to be a node
	tmp = new DecisionNode(*(neg->GetNegative())); 
      }
      // set the covers on the new nodes - SetCovers automatically deletes tCover + vCover
      tmp->SetCovers(tCover, vCover);
      // set the left node
      node->SetNegative(tmp);
      this->nodes += node->GetNegative()->GetNoNodes();
    }
    this->nodes++;
  } else {
    // modify the head
    delete tCover;
    delete vCover;
    // the covers must be all patterns
    tCover = new DynamicArray<Pattern *>(*chromoPool->GetTrainingSet());
    vCover = new DynamicArray<Pattern *>(*data->GetValidationSet());

    if (head->IsConsequent()) {
      delete head;
      // Create new consequent only head, tCover + vCover must not be deleted or segmentation fault
      head = new DecisionNode(new ContinuousConsequent(data, chromoPool, tCover, vCover));
      this->nodes = 1;
      // make a tree quick !
      unsigned rnd = Random::GetRandom(0U, (data->GetDecisionMaxNodes() - 1) / 2);
      for (unsigned i = 0; i < rnd; i++) {
	MutateExpand();
      }
    } else if (Random::GetRandom(0.0, 1.0) < 0.5) {
      // head is antecedent and choose right
      DecisionNode *tmp  = new DecisionNode(*(head->GetAffirmative())); 
      delete head;
      head = tmp;
      // set the covers on the new nodes - SetCovers automatically deletes tCover + vCover
      head->SetCovers(tCover, vCover);
    } else {
      // head is antecedent and choose left
      DecisionNode *tmp = new DecisionNode(*(head->GetNegative())); 
      delete head;
      head = tmp;
      // set the covers on the new nodes - SetCovers automatically deletes tCover + vCover
      head->SetCovers(tCover, vCover);
    }
    this->nodes = head->GetNoNodes();
  }
  // Do not delete tCover + vCover -- otherwise segmentation fault
}

void DecisionTree::MutateNode() {
  DynamicArray<DecisionNode *> nodes;
  DecisionNode *node;
  if (Random::GetRandom(0.0, 1.0) < data->GetDecisionMNAntecedentVsConsequent()) {
    // Mutate an antecedent node
    if (Random::GetRandom(0.0, 1.0) > data->GetDecisionMNWorstVsAnyAntecedent()) {
      // Mutate any antecedent node
      head->ObtainAntecedentNodes(nodes);
      if (nodes.Length() != 0)
	node = nodes[Random::GetRandom(0U, nodes.Length())];
    } else {
      // Mutate the worst antecedent node
      head->ObtainWorstLeafAntecedent(nodes);
      if (nodes.Length() != 0)
	node = nodes[nodes.Length() - 1];
    }
    if (nodes.Length() == 0) {
      // Err - maybe we should try something else
      head->ObtainConsequentNodes(nodes);
    }
    if (nodes.Length() == 0)
      throw new Exception("MutateNode turned up no nodes, odd!");
    node = nodes[Random::GetRandom(0U, nodes.Length())];
  } else {
    // Mutate a consequent node
    if (Random::GetRandom(0.0, 1.0) > data->GetDecisionMNWorstVsAnyConsequent()) {
      // Mutate any consequent
      head->ObtainConsequentNodes(nodes);
      node = nodes[Random::GetRandom(0U, nodes.Length())];
    } else {
      // Mutate the worst consequent
      head->ObtainWorstConsequent(nodes);
      node = nodes[nodes.Length() - 1];
    }
    if (nodes.Length() == 0)
      throw new Exception("MutateNode turned up no nodes, odd!");
  }

  if (node->IsConsequent()) {
    // if the selected node is a consequent, mutate it!
    node->GetConsequent()->Mutate();
  } else {
    // if the selected node is a antecedent, mutate it!
    DynamicArray<Pattern *> *tCover = new DynamicArray<Pattern *>();
    DynamicArray<Pattern *> *vCover = new DynamicArray<Pattern *>();  
    // get the coverage for this node
    node->GetCovers(tCover, vCover);
    node->GetAntecedent()->Mutate();
    // set the new coverage - SetCovers automatically deletes tCover + vCover
    node->SetCovers(tCover, vCover);
    // do not delete tCover + vCover -- they get deleted by setCovers
  }
}

void DecisionTree::Mutate() {
  if (head == NULL)
    throw new Exception("head NULL in DecisionTree, Mutate");
  double rand = Random::GetRandom(0.0, 1.0);
  double me = data->GetDecisionMutateExpand();
  //double mn = GetDecisionMutateNode();
  double ms = data->GetDecisionMutateShrink();
  double mr = data->GetDecisionMutateReinitialize();
  if (rand > mr + ms + me) {
    MutateNode();
  } else if (rand > mr + ms) {
    MutateExpand();
  } else if (rand > mr) {
    MutateShrink();
  } else {
    *this = DecisionTree(data, chromoPool);
  }
}

unsigned DecisionTree::GetNoNodes() {
  return nodes;
}

DecisionTree* DecisionTree::Crossover(DecisionTree &dt) {
  if (head == NULL)
    throw new Exception("head NULL in DecisionTree, Crossover");
  DecisionTree *tmp = new DecisionTree(*this);
  DynamicArray<DecisionNode *> parentA;
  tmp->head->ObtainAntecedentNodes(parentA);

  DynamicArray<DecisionNode *> parentB;
  dt.head->ObtainAllNodes(parentB);

  DynamicArray<Pattern *> *tCover = new DynamicArray<Pattern *>();
  DynamicArray<Pattern *> *vCover = new DynamicArray<Pattern *>();  

  unsigned ra = Random::GetRandom(0U, parentA.Length() + 1);
  unsigned rb = Random::GetRandom(0U, parentB.Length());

  if (parentA.Length() != 0) {
    if (ra == parentA.Length()) {
      delete tCover;
      delete vCover;
      tCover = new DynamicArray<Pattern *>(*chromoPool->GetTrainingSet());
      vCover = new DynamicArray<Pattern *>(*data->GetValidationSet());
      delete tmp->head;
      tmp->head = new DecisionNode(*(parentB[rb]));
      tmp->head->SetCovers(tCover, vCover);
    } else {
      DecisionNode *parA = parentA[ra];
      DecisionNode *parB = new DecisionNode(*(parentB[rb]));
      if (Random::GetRandom(0.0, 1.0) < 0.5) {
	parA->GetAffirmative()->GetCovers(tCover, vCover);
	parA->SetAffirmative(parB);
	parA->GetAffirmative()->SetCovers(tCover, vCover);
      } else {
	parA->GetNegative()->GetCovers(tCover, vCover);
	parA->SetNegative(parB);
	parA->GetNegative()->SetCovers(tCover, vCover);
      }
    }

    tmp->nodes = tmp->head->GetNoNodes();
    while (tmp->nodes > data->GetDecisionMaxNodes())
      tmp->MutateShrink();
    return tmp;

  } else if (parentB.Length() == 1) {
    delete tCover;
    delete vCover;
    delete tmp;
    tmp = new DecisionTree(data, chromoPool);
    return tmp;
  }

  delete tmp;
  delete tCover;
  delete vCover;
  return dt.Crossover(*this);
}

void DecisionTree::Parse() {
  DecisionNode *parse = head->ParseNode();
  if (parse != NULL) {
    parse = new DecisionNode(*parse);
    delete head;
    head = parse;
  }
  this->nodes = head->GetNoNodes();
}

void DecisionTree::Calculate(double tstd, double vstd) {
  if (head == NULL)
    throw new Exception("head NULL in DecisionTree, Calculate");

  double tsse = 0;
  double vsse = 0;
  unsigned complexity = nodes;
  tmae = 0;
  vmae = 0;
  head->CalculateFitness(tsse, vsse, complexity, tmae, vmae);

  unsigned tCover = chromoPool->GetTrainingSet()->Length();
  unsigned vCover = data->GetValidationSet()->Length();

  if (tstd == 0.0) {
    tcd = 0;
    tacd = 0.0;
  } else {
    tcd = 1.0 - tsse / tstd;
    if (tcd < 0)
      tcd = 0;
    if (tcd > 1)
      tcd = 1;
    int l = tCover - complexity;
    if (l <= 0)
      tacd = 0.0;
    else
      tacd = 1.0 - (double) (tCover - 1.0) / (double) l * tsse / tstd;
    if (tacd < 0)
      tacd = 0;
    if (tacd > 1)
      tacd = 1;
  }

  if (vstd == 0.0) {
    vcd = 0;
    vacd = 0.0;
  } else {
    vcd = 1.0 - vsse / vstd;
    if (vcd < 0)
      vcd = 0;
    if (vcd > 1)
      vcd = 1;
    int l = vCover - complexity;
    if (l <= 0)
      vacd = 0.0;
    else
      vacd = 1.0 - (double) (vCover - 1.0) / (double) l * vsse / vstd;
    if (vacd < 0)
      vacd = 0;
    if (vacd > 1)
      vacd = 1;
  }

  tmse = tsse / (double) tCover;
  vmse = vsse / (double) vCover;
  tmae /= (double) tCover;
  vmae /= (double) vCover;
}

double DecisionTree::GetFitness() const {
  return tacd;
}

double DecisionTree::GetTMSE() const {
  return tmse;
}

double DecisionTree::GetVMSE() const {
  return vmse;
}

double DecisionTree::GetTCD() const {
  return tcd;
}

double DecisionTree::GetVCD() const {
  return vcd;
}

double DecisionTree::GetTACD() const {
  return tacd;
}

double DecisionTree::GetVACD() const {
  return vacd;
}

double DecisionTree::GetTMAE() const {
  return tmae;
}

double DecisionTree::GetVMAE() const {
  return vmae;
}

bool DecisionTree::ChromoUsed(Chromo *chromo) {
  if (chromo == NULL)
    new Exception("chromo is NULL in DecisionTree, ChromoUsed");
  if (head == NULL)
    throw new Exception("head NULL in DecisionTree, ChromoUsed");
  return head->ChromoUsed(chromo);
}

void DecisionTree::AddCover(DynamicArray<Pattern *> *tCover) {
  if (tCover->Length() == 0)
    return;
  DynamicArray<Pattern *> *t = new DynamicArray<Pattern *>(*tCover);
  head->AddCover(t);
}

void DecisionTree::SetCovers(DynamicArray<Pattern *> *tCover, DynamicArray<Pattern *> * vCover) {
  head->SetCovers(new DynamicArray<Pattern *>(*tCover), new DynamicArray<Pattern *>(*vCover));
}

void DecisionTree::CalculateGeneralization(DynamicArray<Pattern *> *gCover, double std) {
  double gsse = 0;
  unsigned complexity = nodes;
  gmae = 0;
  head->CalculateGeneralization(new DynamicArray<Pattern *>(*gCover), gsse, complexity, gmae);

  cout << "STD " << std << endl;
  cout << "SSE " << gsse << endl;
  cout << "COM " << complexity << endl;

  unsigned length = gCover->Length();
  cout << "LEN " << length << endl;

  if (std == 0.0) {
    gcd = 0;
    gacd = 0.0;
  } else {
    gcd = 1.0 - gsse / std;
    if (gcd < 0)
      gcd = 0;
    if (gcd > 1)
      gcd = 1;
    int l = length - complexity;
    if (l <= 0)
      gacd = 0.0;
    else
      gacd = 1.0 - (double) (length - 1.0) / (double) l * gsse / std;
    if (gacd < 0)
      gacd = 0;
    if (gacd > 1)
      gacd = 1;
  }
  gmse = gsse / (double) length;
  gmae /= (double) length;
}

double DecisionTree::GetGMSE() const {
  return gmse;
}

double DecisionTree::GetGCD() const {
  return gcd;
}

double DecisionTree::GetGACD() const {
  return gacd;
}

double DecisionTree::GetGMAE() const {
  return gmae;
}

void DecisionTree::Optimize() {
  head->Optimize();
}

void DecisionTree::CalculateMiscellaneous(double &terms, double &rules, double &conditions) {
  unsigned t = 0;
  unsigned r = 0;
  unsigned c = 0;
  head->CalculateMiscellaneous(r, c, t, 0);
  terms = ((double) t) / r;
  rules = r;
  conditions = ((double) c) / r;
}
