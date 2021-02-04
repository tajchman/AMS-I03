#pragma once

class VectorCUDA {

public:

  VectorCUDA() : m_n(0), m_c(0) {}
  ~VectorCUDA();

  void resize(size_t n);
  void zero();

private:
  size_t m_n;
  double * m_c;
};
  
class MatrixCUDA {

public:

  MatrixCUDA() : m_n(0), m_m(0), m_c(0) {};
  ~MatrixCUDA();

  void resize(size_t n, size_t m);
  void zero();
  

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

private:
  
  std::vector<size_t> m_sizes;
  std::vector<VectorCUDA> m_v;
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

private:
  
  std::vector<size_t> m_sizes;
  std::vector<MatrixCUDA> m_m;
};
