#pragma once

#include <vector>

class Vector {

public:

  Vector() : m_n(0), m_c(0) {}
  ~Vector();
  Vector(const std::vector<double> & v);
  void operator=(const std::vector<double> & v);
  void resize(size_t n);
  void zero();
  void operator += (const Vector& v);

private:
  size_t m_n;
  double * m_c;
};
  
class Matrix {

public:

  Matrix() : m_n(0), m_m(0), m_c(0) {};
  ~Matrix();

  void resize(size_t n, size_t m);
  void zero();
  void operator += (const Matrix& v);

private:
  size_t m_n, m_m;
  double * m_c;
};

class BiasesVector {

public:
  
  BiasesVector(const std::vector<size_t> &sizes)
    : m_sizes(sizes), m_v(sizes.size()-1) {
    size_t i;
    for (i=0; i<m_v.size(); i++)
      m_v[i].resize(m_sizes[i+1]);
  }

  void zero();
  size_t size() { return m_v.size(); }
  Vector & operator[] (size_t i) { return m_v[i]; }

private:
  
  std::vector<size_t> m_sizes;
  std::vector<Vector> m_v;
};

class WeightsVector {

public:
  
  WeightsVector(const std::vector<size_t> &sizes)
    : m_sizes(sizes), m_m(sizes.size()-1) {
    size_t i;
    for (i=0; i<m_m.size(); i++)
      m_m[i].resize(m_sizes[i], m_sizes[i+1]);
  }

  void zero();
  Matrix & operator[] (size_t i) { return m_m[i]; }

private:
  
  std::vector<size_t> m_sizes;
  std::vector<Matrix> m_m;
};

using DataCPU = std::pair<std::vector<double>, std::vector<double>>;
using DataCPUIterator = typename std::vector<DataCPU>::iterator;

using DataGPU = std::pair<Vector, Vector>;
using DataGPUIterator = typename std::vector<DataGPU>::iterator;


