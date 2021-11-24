#ifndef PATTERN_LIST_H
#define PATTERN_LIST_H

#include <iostream>
#include <cstddef>
#include <cstdlib>
#include "Pattern.h"
#include "../templates/Exceptions.h"

class PatternNode {
 public:
  Pattern *pattern;
  PatternNode *next;
  PatternNode *prev;

  PatternNode(Pattern *);
  PatternNode(Pattern *, PatternNode *, PatternNode *);
  ~PatternNode();

  friend ostream& operator << (ostream &, const PatternNode &);
  friend ostream& operator << (ostream &, const PatternNode *);
};

class PatternList {
 public:
  PatternList();
  ~PatternList();

  unsigned Length() const;
  void ToFront();
  void ToBack();
  bool Next();
  bool Prev();
  void AddPattern(Pattern *);
  void RemovePattern();
  Pattern* GetPattern();
  Pattern* operator [] (unsigned) const;
  void Finalize();

 private:
  PatternNode *head;
  PatternNode *tail;
  PatternNode *current;
  Pattern **array;
  unsigned noPatterns;
  bool finalized;
  bool deleted;

  friend ostream& operator << (ostream &, PatternList &);
};

#endif
