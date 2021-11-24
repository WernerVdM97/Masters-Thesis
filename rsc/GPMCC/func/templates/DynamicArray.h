#ifndef DYNA_H
#define DYNA_H
#include <iostream>
#include "Exceptions.h"

template <class T> 
class DynamicArray {
 public:
  DynamicArray();
  DynamicArray(unsigned);
  DynamicArray(const DynamicArray<T> &);
  ~DynamicArray();

  unsigned Length() const;
  T& operator [] (unsigned) const;
  void operator = (const DynamicArray<T>&);

  void Add(const T&);
  void Update(const T&, unsigned);
  void Delete(unsigned);
  void Insert(const T&, unsigned);
  void InsertSorted(const T&);
  bool InsertUniqueSorted(const T&);
  int BinarySearch(const T&) const;
  void Sort();

 private:
  T *array;
  unsigned size;
  unsigned length;

  void IncreaseCapacity();
  void QSort(int,int);

  friend ostream& operator <<<T> (ostream &os, DynamicArray<T> &array);
};

#include "DynamicArray.C"
#endif
