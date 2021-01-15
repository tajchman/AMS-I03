#include "matrice.hxx"
#include "timer.hxx"
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
    bytes = m_nm * sizeof(double);
    cudaMalloc(&(d_c), bytes);

    T.stop();
}

Matrice::~Matrice() {
  Timer & T = GetTimer(0);
  T.start();

  delete [] h_c;
  cudaFree(d_c);

  T.stop();
}

Matrice::Matrice(const Matrice &other) 
    : m_n(other.m_n), 
      m_m(other.m_m), 
      m_nm(other.m_nm), 
      bytes(other.bytes),
      m_name(other.m_name)
{
    Timer & T0 = GetTimer(0);
    T0.start();

    h_c = new double[other.m_nm];
    cudaMalloc(&(d_c), bytes);

    T0.stop();

    Timer & T1 = GetTimer(1);
    T1.start();

    std::memcpy(h_c, other.h_c, bytes);
    cudaMemcpy (d_c, other.d_c, bytes, cudaMemcpyDeviceToDevice);
    m_synchronized = other.m_synchronized;
    
    T1.stop();
}

void Matrice::copyToDevice()
{
  cudaMemcpy (d_c, h_c, bytes, cudaMemcpyHostToDevice);
  m_synchronized = true;
}

void Matrice::copyToHost()
{
  cudaMemcpy (h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  m_synchronized = true;
}

void Matrice::synchronize()
{
   if (m_synchronized) return;
   copyToHost();
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
        cudaFree(d_c);
        cudaMalloc(&(d_c), bytes);

        T.stop();
    }
    Timer & T = GetTimer(1);
    T.start();

    m_n = other.m_n;
    m_m = other.m_m;
    cudaMemcpy (d_c, other.d_c, bytes, cudaMemcpyDeviceToDevice);
    m_synchronized = other.m_synchronized;
    if (m_synchronized)
      std::memcpy(h_c, other.h_c, m_nm*sizeof(double));

    T.stop();
}

__global__ void initGPU(double *c, double v, int n) {
    int i;
    i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) {
      c[i] = v;
    }
  }
    
void Matrice::Init(double v) {
   int blockSize = 256;
   int gridSize = (m_nm + blockSize)/blockSize;
   initGPU<<<gridSize, blockSize>>>(d_c, v, m_nm);
   m_synchronized = false;
}

__global__ void identiteGPU(double *c, int n) {
    int i, j;
    i = blockIdx.x*blockDim.x+threadIdx.x;
    j = blockIdx.y*blockDim.y+threadIdx.y;
    if (i < n && i == j) {
      c[i + n*j] = 1.0;
    }
  }

void Matrice::Identite() {

    Timer & T = GetTimer(2);
    T.start();

    Init(0.0);

    dim3 blockSize(16, 16);
    dim3 gridSize((m_n + blockSize.x)/blockSize.x,
                  (m_m + blockSize.y)/blockSize.y); 
      
    identiteGPU<<<gridSize, blockSize>>>(d_c, m_n);
    m_synchronized = false;

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

    Timer & T1 = GetTimer(1);
    T1.start();

    copyToDevice();

    T1.stop();
}

inline double sqr(double x) { return x*x;}

double Matrice::norm2() {

    Timer & T0 = GetTimer(1);
    T0.start();
    
    copyToHost();

    T0.stop();
   
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

__global__ void raddGPU(double *v, double *u, int n) {
    int i;
    i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) {
      v[i] += u[i];
    }
  }
    
void Matrice::operator+=(const Matrice &M)
{
    Timer & T = GetTimer(5);
    T.start();

    int blockSize = 256;
    int gridSize = (m_nm + blockSize)/blockSize;
    raddGPU<<<gridSize, blockSize>>>(d_c, M.d_c, m_nm);
    m_synchronized = false;

    T.stop();       
}

__global__ void rsubGPU(double *v, double *u, int n) {
    int i;
    i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) {
      v[i] -= u[i];
    }
  }
    
void Matrice::operator-=(const Matrice &M)
{
    Timer & T = GetTimer(5);
    T.start();

    int blockSize = 256;
    int gridSize = (m_nm + blockSize)/blockSize;
    rsubGPU<<<gridSize, blockSize>>>(d_c, M.d_c, m_nm);
    m_synchronized = false;

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
    Timer & T = GetTimer(6);
    T.start();

    int n = M2.n(), p = M2.m(), q = M3.n(), m = M3.m();
    if (p != q || n != M1.n() || m != M1.m())
        throw std::runtime_error("erreur de dimensions");

    double *d1 = M1.device();
    const double *d2 = M2.device(), *d3 = M3.device();

    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x)/blockSize.x,
                  (m + blockSize.y)/blockSize.y); 
    
    multiplyGPU<<<gridSize, blockSize>>>(d1, d2, d3, n, p, m);

    T.stop();
}

#include <iomanip>

std::ostream & operator << (std::ostream & f, Matrice & M)
{
    M.synchronize();

    f << std::endl << M.name() << std::endl;
    for (int i=0; i<M.n(); i++) {
        f << std::setw(4) << i << ": ";
        for (int j=0; j<M.m(); j++)
            f << std::setw(14) << M(i,j);
        f << std::endl;
    }
    return f;
}
