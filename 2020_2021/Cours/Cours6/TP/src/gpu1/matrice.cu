#include "matrice.hxx"
#include "timer.hxx"
#include "cuda_check.cuh"
#include <cstring>
#include <cstdlib>
#include <stdexcept>

Matrice::Matrice(int n, int m, const char * name) 
    : m_n(n), m_m(m), m_nm(n*m), 
      m_name(name)
{
    Timer & T = GetTimer(0);
    T.start();

    h_c = new double[m_nm];

    T.stop();
}

Matrice::~Matrice() {
    delete [] h_c;
}

Matrice::Matrice(const Matrice &other) 
    : m_n(other.m_n), 
      m_m(other.m_m), 
      m_nm(other.m_nm), 
      m_name(other.m_name)
{
    Timer & T0 = GetTimer(0);
    T0.start();
    h_c = new double[other.m_nm];
    T0.stop();

    Timer & T1 = GetTimer(1);
    T1.start();
    std::memcpy(h_c, other.h_c, m_nm * sizeof(double));
    T1.stop();
}

void Matrice::operator=(const Matrice &other)
{
    if (&other == this)
        return;

    m_n = other.m_n;
    m_m = other.m_m;
    if (m_nm != other.m_nm) {
        Timer & T = GetTimer(0);
        T.start();

        delete [] h_c;
        h_c = new double[other.m_nm];

        T.stop();
    }
    Timer & T = GetTimer(1);
    T.start();

    m_n = other.m_n;
    m_m = other.m_m;
    std::memcpy(h_c, other.h_c, m_nm*sizeof(double));

    T.stop();
}

void Matrice::Init(double v) {
    int i;

    for (i=0; i<m_nm; i++)
        h_c[i] = v;
}


void Matrice::Identite() {

    Timer & T = GetTimer(2);
    T.start();

    Init(0.0);

    for (int i=0; i<m_n;i++)
        (*this)(i,i) = 1.0;

    T.stop();
}

void Matrice::Random(double cmax) {

    Timer & T = GetTimer(4);
    T.start();

    std::srand(0);
    int i,j;
    for (i=0; i<m_n;i++)
        for (j=0; j<m_m; j++) {
            double c = (cmax * std::rand())/RAND_MAX;
            (*this)(i,j) = c;
        }
    for (i=0; i<m_n;i++)
       (*this)(i,i) += 1.0;

    T.stop();
}

inline double sqr(double x) { return x*x;}

double Matrice::norm2() {

    Timer & T = GetTimer(3);
    T.start();

    int i,j;
    double s = 0.0;
    for (i=0; i<m_n;i++)
        for (j=0; j<m_m; j++)
            s += sqr((*this)(i,j));
    s = sqrt(s);

    T.stop();
    return s;
}

void Matrice::operator+=(const Matrice &M)
{
    Timer & T = GetTimer(5);
    T.start();

    int i;
    for (i=0;i<m_nm;i++)
      h_c[i] += M.h_c[i];

    T.stop();       
}

void Matrice::operator-=(const Matrice &M)
{
    Timer & T = GetTimer(5);
    T.start();

    int i;
    for (i=0;i<m_nm;i++)
      h_c[i] -= M.h_c[i]; 

    T.stop();       
}

__global__ void multiplyGPU(double *w, const double *u, const double *v,
    int n, int p, int m)
{
  int i, j;
  i = blockIdx.x*blockDim.x+threadIdx.x;
  j = blockIdx.y*blockDim.y+threadIdx.y;

  if (i >= n || j >= m) return;

  double s = 0.0;
  int k;

  for (k=0; k<p; k++)
    s += u[k + i*p] * v[j + k*m];
  w[j+i*m] = s;
}

void multiply(Matrice & M1, const Matrice &M2, const Matrice &M3)
{
    int n = M2.n(), p = M2.m(), q = M3.n(), m = M3.m();
    if (p != q || n != M1.n() || m != M1.m())
        throw std::runtime_error("erreur de dimensions");

    Timer & T1 = GetTimer(0);
    T1.start();

    double *d1, *d2, *d3;

    size_t bytes1 = M1.n() * M1.m() * sizeof(double);
    cudaMalloc(&(d1), bytes1);

    size_t bytes2 = M2.n() * M2.m() * sizeof(double);
    cudaMalloc(&(d2), bytes2);

    size_t bytes3 = M3.n() * M3.m() * sizeof(double);
    cudaMalloc(&(d3), bytes3);

    T1.stop();

    Timer & T2 = GetTimer(1);
    T2.start();

    cudaMemcpy (d1, M1.data(), bytes1, cudaMemcpyHostToDevice);
    cudaMemcpy (d2, M2.data(), bytes2, cudaMemcpyHostToDevice);
    cudaMemcpy (d3, M3.data(), bytes3, cudaMemcpyHostToDevice);

    T2.stop();

    Timer & T = GetTimer(6);
    T.start();
    
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x)/blockSize.x,
                  (m + blockSize.y)/blockSize.y); 
    
    multiplyGPU<<<gridSize, blockSize>>>(d1, d2, d3, n, p, m);

    T.stop();
 
    T2.start();

    cudaMemcpy (M1.data(), d1, bytes1, cudaMemcpyDeviceToHost);

    T2.stop();

    T1.start();

    cudaFree(d3);
    cudaFree(d2);
    cudaFree(d1);

    T1.start();
}

#include <iomanip>

std::ostream & operator << (std::ostream & f, Matrice & M)
{
    f << std::endl << M.name() << std::endl;
    for (int i=0; i<M.n(); i++) {
        f << std::setw(4) << i << ": ";
        for (int j=0; j<M.m(); j++)
            f << std::setw(14) << M(i,j);
        f << std::endl;
    }
    return f;
}
