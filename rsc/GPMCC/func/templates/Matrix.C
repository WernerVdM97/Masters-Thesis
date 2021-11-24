/* Class: Matrix
   By:    G. Potgieter

   Do you need linear algerbra? Then this class is for you!!!

   FEATURES
   - Fast, Accurate Row-Echelon reduction.
   - Fast Gauss-Jordan reduction.
   - Fast Determinant calculation.

   ALLOWS
   - Fast, accurate solving of linear systems.
   - Transposition
   - Matrix multiplication

   Note to self: Get a copy of numerical recipies.
*/

#include "Matrix.h"

Matrix::Matrix(unsigned m) {
  if (m < 1)
    return;
  across = m;
  down = m;
  matrix = new double*[down];
  for (unsigned i = 0; i < down; i++)
    matrix[i] = new double[across];
  cAcross = 0;
  cDown = 0;
  determinant = 1.0;
}

Matrix::Matrix(unsigned n, unsigned m) {
  if ((m < 1) || (n < 1))
    return;
  across = m;
  down = n;
  matrix = new double*[down];
  for (unsigned i = 0; i < down; i++)
    matrix[i] = new double[across];
  cAcross = 0;
  cDown = 0;
  determinant = 1.0;
}

Matrix::~Matrix() {
  for (unsigned i = 0; i < down; i++)
    delete [] matrix[i];
  delete [] matrix;
}

void Matrix::RowEchelonForm() {
  unsigned in = 0;
  for (unsigned i = 0; i < down; i++) {
    if (in == across)
      break;
    if (matrix[i][in] == 0.0) {
      unsigned find = i;
      for (unsigned j = i + 1; j < down; j++)
        if (matrix[j][in] /*!= 0.0*/ > matrix[find][in]) {
          find = j;
          //break;
	}
      if (find != i) {
        double *tmp = matrix[i];
        matrix[i] = matrix[find];
        matrix[find] = tmp;
        determinant *= -1;
      } else {
        determinant *= 0;
        i--;
        in++;
        continue;
      }
    }

    determinant *= matrix[i][in];
    for (unsigned k = across; k > in; k--) {   
      matrix[i][k-1] = matrix[i][k-1] / matrix[i][in];
    }

    for (unsigned j = i + 1; j < down; j++) {
      for (unsigned k = across; k > in; k--) {   
	matrix[j][k-1] = -matrix[j][in]*matrix[i][k-1] + matrix[j][k-1];
      }
    }
    in++;
  }
}

void Matrix::GaussJordanForm() {
  RowEchelonForm();
  unsigned smaller = (down < across)?down:across;
  unsigned in = smaller - 1;
  unsigned k;

  for (unsigned i = smaller - 1; i > 0; i--) {
    for (unsigned j = 0; j < i; j++) {
      if (matrix[j][in] != 0.0) {
        for (k = down; k < across; k++) {
	  matrix[j][k] -= matrix[j][in] * matrix[i][k];
	}

        k = in;
	matrix[j][k] -= matrix[j][in] * matrix[i][k];
      }
    }
    in--;
  }
}

void Matrix::Identity() {
  if (across != down)
    throw new MatrixException("matrix is non-square");
  for (unsigned i = 0; i < down; i++)
    for (unsigned j = 0; j < across; j++)
      if (i != j)
	matrix[i][j] = 0;
      else
	matrix[i][j] = 1;
}

void Matrix::Truncate() {
  for (unsigned i = 0; i < down; i++)
    for (unsigned j = 0; j < across; j++) {
      if ((matrix[i][j] < 0.00000000001) && (matrix[i][j] > -0.00000000001))
	matrix[i][j] = 0;
    }
}

double Matrix::Determinant() {
  return determinant;
}

void Matrix::Insert(double val) {
  matrix[cDown][cAcross] = val;
  cAcross++;
  if (cAcross == across) {
    cAcross = 0;
    cDown++;
  }  
  if (cDown == down) {
    cDown = 0;
  }
}

double Matrix::Retrieve() {
  double val = matrix[cDown][cAcross];
  cAcross++;
  if (cAcross == across) {
    cAcross = 0;
    cDown++;
  }  
  if (cDown == down) {
    cDown = 0;
  }
  return val;
}

Matrix* Matrix::Transpose() {
  Matrix *transpose = new Matrix(across,down);
  for (unsigned i = 0; i < down; i++) {
    for (unsigned j = 0; j < across; j++) {
      transpose->matrix[j][i] = matrix[i][j];
    }
  }
  return transpose;
}

Matrix* Matrix::Multiply(Matrix *mat) {
  if (mat == NULL)
    return NULL;
  if (across != mat->down)
    return NULL;
  Matrix *tmp = new Matrix(down, mat->across);
  double total = 0.0;
  for (unsigned i = 0; i < down; i++) {
    for (unsigned j = 0; j < mat->across; j++) {
      total = 0.0;
      for (unsigned k = 0; k < across; k++)
        total += matrix[i][k] * mat->matrix[k][j];
      tmp->matrix[i][j] = total;
    }
  }
  tmp->Truncate();
  return tmp;
}

Matrix *Matrix::Solve(Matrix *vec) {
  if (vec == NULL)
    return NULL;
  if ((vec->across != 1) && (vec->across != across) && (vec->down != down) && (across != down))
    return NULL;

  Matrix *tmp = new Matrix(down, across + vec->across);
  for (unsigned i = 0; i < down; i++) {
    for (unsigned j = 0; j < across; j++)
      tmp->matrix[i][j] = matrix[i][j];
    for (unsigned j = across; j < across + vec->across; j++) {
      tmp->matrix[i][j] = vec->matrix[i][j - across];
    }
  }

  tmp->GaussJordanForm();
  tmp->Truncate();
  Matrix *tmp2 = new Matrix(down, vec->across);
  for (unsigned i = 0; i < down; i++) {
    for (unsigned j = across; j < across + vec->across; j++) {
      tmp2->matrix[i][j - across] = tmp->matrix[i][j];
    }
  }
  delete tmp;
  return tmp2;
}

ostream& operator << (ostream &os, const Matrix &matrix) {
  for (unsigned i = 0; i < matrix.down; i++) {
    for (unsigned j = 0; j < matrix.across; j++) {
      os << matrix.matrix[i][j];
      if (j != matrix.across - 1)
        os << "\t";
    }
    os << endl;
  }
  return os;
}



