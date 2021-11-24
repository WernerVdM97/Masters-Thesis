#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <string>
#include <iostream>

using namespace std;

class Exception {
 public:
  Exception(const string &error) {
    this->error = error;
    //cout << *this << endl;
    //exit(-1);
  }
  virtual ~Exception() {}
  virtual string GetMessage() const {
    return "Exception:" + error;
  }
  friend ostream& operator << (ostream &os, const Exception &exception) {
    os << exception.GetMessage();
    return os;
  }
 protected:
  string error;
};

class ParseException : public Exception {
 public:
  ParseException(const string &error) : Exception(error) {}
  virtual ~ParseException() {}
  virtual string GetMessage() const {
    return "ParseException: " + error;
  }
};

class FileException : public Exception {
 public:
  FileException(const string &error) : Exception(error) {}
  virtual ~FileException() {}
  virtual string GetMessage() const {
    return "FileException: " + error;
  }
};

class IndexOutOfBoundsException : public Exception {
 public:
  IndexOutOfBoundsException() : Exception("Index out of bounds") {}
  virtual ~IndexOutOfBoundsException() {}
  virtual string GetMessage() const {
    return "IndexOutOfBoundsException: Index out of bounds";
  }
};

class NullPointerException : public Exception {
 public:
  NullPointerException() : Exception("Null pointer") {}
  virtual ~NullPointerException() {}
  virtual string GetMessage() const {
    return "NullPointerException: Null pointer";
  }
};

class UnmatchedLengthException : public Exception {
 public:
  UnmatchedLengthException() : Exception("Lengths do not match") {}
  virtual ~UnmatchedLengthException() {}
  virtual string GetMessage() const {
    return "UnmatchedLengthException: Lengths do not match";
  }
};

class MatrixException : public Exception {
 public:
  MatrixException(string error) : Exception(error) {}
  virtual ~MatrixException() {}
  virtual string GetMessage() const {
    return "MatrixException: " + error;
  }
};

#endif
