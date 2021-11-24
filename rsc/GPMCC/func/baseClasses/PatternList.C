/* Class: PatternList
   By G. Potgieter

   A variable length linked list class that features:
   - Fast inserts, O(1)
   - Enumeration
   - Bidirectional operation
   - Finalization to an array for fast retrievals.
*/

#include "PatternList.h"

PatternNode::PatternNode(Pattern *pattern) {
  this->pattern = pattern;
  this->next = NULL;
  this->prev = NULL;
}

PatternNode::PatternNode(Pattern *pattern, PatternNode *next, PatternNode *prev) {
  //cout << "<node> " << ((unsigned) this) << endl;
  this->pattern = pattern;
  this->next = next;
  this->prev = prev;
}

PatternNode::~PatternNode() {
  //cout << "~<node> " << ((unsigned) this) << endl;
  next = NULL;
  prev = NULL;
}

PatternList::PatternList() {
  head = NULL;
  tail = NULL;
  current = NULL;
  noPatterns = 0;
  finalized = false;
  array = NULL;
  deleted = false;
}

PatternList::~PatternList() {
  if (!finalized) {
    ToFront();
    while (current != NULL) {
      RemovePattern();
      Next();
    }
  } else {
    delete [] array;
  }
}

unsigned PatternList::Length() const {
  return noPatterns;
}

void PatternList::AddPattern(Pattern *pattern) {
  PatternNode *tmp = new PatternNode(pattern, head, NULL);
  if (head != NULL)
    head->prev = tmp;
  else
    tail = tmp;
  head = tmp;
  noPatterns++;
}

void PatternList::ToFront() {
  current = head;
  deleted = false;
}

void PatternList::ToBack() {
  current = tail;
  deleted = false;
}

bool PatternList::Next() {
  if ((current != NULL) && (!deleted)) {
    current = current->next;
  } else if (deleted) {
    deleted = false;
  }
  return (current != NULL);
}

bool PatternList::Prev() {
  if (current != NULL)
    current = current->prev;
  return (current != NULL);
}

Pattern* PatternList::GetPattern() {
  if (current == NULL)
    throw new NullPointerException();
  return current->pattern;
}

void PatternList::RemovePattern() {
  if (current == NULL)
    throw new NullPointerException();
  PatternNode *tmp = current;
  current = current->next;
  deleted = true;
  
  if (tmp != head) {
    (tmp->prev)->next = tmp->next;
  } else {
    head = tmp->next;
  }
  
  if (tmp != tail) {
    (tmp->next)->prev = tmp->prev;
  } else {
    tail = tmp->prev;
  }
    
  delete tmp;
  noPatterns--;
}

void PatternList::Finalize() {
  if (!finalized) {
    array = new Pattern*[noPatterns];
    ToFront();
    int i = 0;
    while (current != NULL) {
      array[i] = GetPattern();
      RemovePattern();
      i++;
      Next();
    }
    noPatterns = i;
    finalized = true;
  }
}


Pattern* PatternList::operator [] (unsigned index) const {
  if ((!finalized) || (array == NULL) || (index >= noPatterns))
    throw new IndexOutOfBoundsException();
  return array[index];
}

ostream& operator << (ostream &os, PatternList &pl) {
  if (!pl.finalized) {
    PatternNode *tmp = pl.head;
    while (tmp != NULL) {
      os << tmp << endl;
      tmp = tmp->next;
    }
  } else {
    for (unsigned i = 0; i < pl.noPatterns; i++) {
      os << pl[i] << endl;
    }
  }
  return os;
}

ostream& operator << (ostream &os, const PatternNode &pn) {
  os << &pn;
  return os;
}

ostream& operator << (ostream &os, const PatternNode *pn) {
  if (pn == NULL)
    os << "null";
  else {
    if (pn->pattern != NULL)
      os << pn->pattern;
    else
      os << "null";
    if (pn->next != NULL)
      os << "(" << pn->next->pattern << ",";
    else
      os << "(null,";
    if (pn->prev != NULL)
      os << pn->prev->pattern << ")";
    else
      os << "null)";
  }
  return os;
}
