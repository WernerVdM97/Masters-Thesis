/* Class: DynamicArray
   By:    G. Potgieter

   Do you sometimes need a good, fast, javaesq Vector class under C++?
   Well, your wishes have come true!!!

   There are three modes of operation.
   - Variable length array.
   - Variable length sorted array.
   - Variable length unique sorted array.

   All sorting done with Quicksort, and binary search.
   - inserts cost N if unsorted, NlogN if sorted -- not bad.
   - searching logN -- cool!
   - sorting if unsorted NlogN -- way to go!
*/

template <class T>
DynamicArray<T>::DynamicArray() {
  size = 10;
  length = 0;
  array = new T[10];
  //cout << "+DynamicArray: " << this << endl;
}

template <class T>
DynamicArray<T>::DynamicArray(unsigned size) {
  this->size = size;
  length = 0;
  array = new T[size];
  //cout << "+DynamicArray: " << this << endl;
}

template <class T>
DynamicArray<T>::DynamicArray(const DynamicArray<T> &make) {
  size = make.size;
  length = make.length;
  array = new T[size];
  for (unsigned i = 0; i < size; i++)
    array[i] = make.array[i];
  //cout << "+DynamicArray: " << this << endl;
}

template <class T>
DynamicArray<T>::~DynamicArray() {
  delete [] array;
  //cout << "~DynamicArray: " << this << endl;
}

template <class T>
unsigned DynamicArray<T>::Length() const {
  return length;
}

template <class T>
void DynamicArray<T>::IncreaseCapacity() {
  unsigned tSize = size / 10 + size;
  if (tSize == size)
    tSize++;

  T *tmp = new T[tSize];
  for (unsigned i = 0; i < length; i++)
    tmp[i] = array[i];
  delete [] array;
  array = tmp;
  size = tSize;
}

template <class T>
T& DynamicArray<T>::operator [] (unsigned index) const {
  if (index > length)
    throw new IndexOutOfBoundsException();
  return array[index];
}

template <class T>
void DynamicArray<T>::Update(const T& item, unsigned index) {
  if (index >= length)
    throw new IndexOutOfBoundsException();
  array[index] = item;
  return;
}

template <class T>
void DynamicArray<T>::Add (const T& item) {
  array[length] = item;
  length++;
  if (length >= size)
    IncreaseCapacity();
}

template <class T>
void DynamicArray<T>::Delete (unsigned index) {
  if ((index >= length) || (length == 0))
    throw new IndexOutOfBoundsException();

  for (unsigned i = index; i < length - 1; i++)
    array[i] = array[i+1];
  length--;
}

template <class T>
void DynamicArray<T>::Insert (const T& item, unsigned index) {
  if (index > length)
    throw new IndexOutOfBoundsException();
  length++;
  if (length >= size)
    IncreaseCapacity();

  for (unsigned i = length - 1; i > index; i--)
    array[i] = array[i-1];
  array[index] = item;
}

template <class T>
int DynamicArray<T>::BinarySearch (const T& item) const {
  if (length == 0)
    return -1;
  unsigned i = 0;
  unsigned j = length - 1;
  unsigned pivot;

  while (i < j) {
    pivot = (i + j + 1) / 2;
    if (item == array[pivot]) {
      return pivot;
    } else if (item >/*< asc*/ array[pivot]) {
      j = pivot - 1;
    } else {
      i = pivot + 1;
    }
  }
  if (i >= length)
    i = length - 1;
  if (array[i] </*> asc*/ item)
    return -i - 1;
  else if (array[i] >/*< asc*/ item)
    return -i - 2;
  else
    return i;
}

template <class T>
ostream& operator << (ostream &os, DynamicArray<T> &array) {
  for (unsigned i = 0; i < array.length; i++)
    os << array.array[i] << " ";
  return os;
}

template <class T>
void DynamicArray<T>::InsertSorted (const T& item) {
  int found = BinarySearch(item);
  if (found < 0) {
    found = -found - 1;
  }
  Insert(item, found);
}

template <class T>
bool DynamicArray<T>::InsertUniqueSorted (const T& item) {
  int found = BinarySearch(item);
  if (found < 0) {
    found = -found - 1;
    Insert(item, found);
    return true;
  }
  return false;
}

template <class T>
void DynamicArray<T>::Sort() {
  QSort(0, length-1);
}

template <class T>
void DynamicArray<T>::QSort(int lower, int upper) {
  // first time upper = n - 1;
  // first time lower = 0;
  int i = lower;
  int j = upper;
  T pivot = array[(lower+upper)/2];
  if (lower >= upper)
    return;
  do {
    while (array[i] >/*< asc*/ pivot)
      i++;
    while (pivot >/*< asc*/ array[j])
      j--;
    if (i <= j) {
      T tmp = array[i];
      array[i++] = array[j];
      array[j--] = tmp;
    }      
  } while (i <= j);
  QSort(lower, j);
  QSort(i, upper);
}

template <class T>
void DynamicArray<T>::operator = (const DynamicArray<T>& make) {
  if (this == &make)
    return;
  size = make.size;
  length = make.length;
  delete [] array;
  array = new T[size];
  for (unsigned i = 0; i < size; i++)
    array[i] = make.array[i];
}
