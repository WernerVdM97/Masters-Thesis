#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include "Exceptions.h"

class Matrix {
  public:
    Matrix(unsigned);
    Matrix(unsigned, unsigned);
    ~Matrix();

    void RowEchelonForm();
    void GaussJordanForm();
    double Determinant();

    void Insert(double);
    double Retrieve();
    void Identity();

    Matrix *Transpose();
    Matrix *Multiply(Matrix *);
    Matrix *Solve(Matrix *);

 private:
    double **matrix;
    unsigned across;
    unsigned down;

    unsigned cAcross;
    unsigned cDown;
    double determinant;

    void Truncate();

    friend ostream& operator << (ostream &, const Matrix &);
};

#endif
