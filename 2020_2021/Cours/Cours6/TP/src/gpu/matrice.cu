#include "matrice.hxx"
#include <cstring>
#include <cstdlib>
#include <stdexcept>

Matrice::Matrice(int n, int m, const char * name) 
    : m_n(n), m_m(m), m_nm(n*m), 
      h_c(new double[n*m]),
      m_name(name)
{
    bytes = n * m * sizeof(double);
    cudaMalloc(&(d_c), bytes);
}

Matrice::~Matrice() {
    cudaFree(d_c);
}

Matrice::Matrice(const Matrice &other) 
    : m_n(other.m_n), 
      m_m(other.m_m), 
      m_nm(other.m_nm), 
      bytes(other.bytes),
      m_name(other.m_name)
{
    h_c = new double[other.m_nm];
    std::memcpy(h_c, other.h_c, bytes);
    cudaMalloc(&(d_c), bytes);
    cudaMemcpy (d_c, other.d_c, bytes, cudaMemcpyDeviceToDevice);
    m_synchronized = other.m_synchronized;
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

void Matrice::operator=(const Matrice &other)
{
    if (&other == this)
        return;

    m_n = other.m_n;
    m_m = other.m_m;
    if (m_nm != other.m_nm) {
        delete [] h_c;
        h_c = new double[other.m_nm];
        cudaFree(d_c);
        cudaMalloc(&(d_c), bytes);
    }
    m_n = other.m_n;
    m_m = other.m_m;
    cudaMemcpy (d_c, other.d_c, bytes, cudaMemcpyDeviceToDevice);
    m_synchronized = other.m_synchronized;
    if (m_synchronized)
      std::memcpy(h_c, other.h_c, m_nm*sizeof(double));
      
}

void Matrice::synchronize()
{
   if (m_synchronized) return;
   copyToHost();
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

    Init(0.0);

    dim3 blockSize(16, 16);
    dim3 gridSize((m_n + blockSize.x)/blockSize.x,
                  (m_m + blockSize.y)/blockSize.y); 
      
    identiteGPU<<<gridSize, blockSize>>>(d_c, m_n);
}

void Matrice::Random(double cmax) {

    int i,j;
    for (i=0; i<m_n;i++)
        for (j=0; j<m_m; j++) {
            double c = (cmax * std::rand())/RAND_MAX;
            (*this)(i,j) = c;
        }
    for (i=0; i<m_n;i++)
       (*this)(i,i) += 1.0;
    
    copyToDevice();
}

inline double sqr(double x) { return x*x; }
double Matrice::norm2() {

    copyToHost();
   
    int i,j;
    double s = 0.0;
    for (i=0; i<m_n;i++)
        for (j=0; j<m_m; j++)
            s += sqr((*this)(i,j));

    copyToDevice();

    return sqrt(s);
}

void Matrice::operator+=(const Matrice &M)
{
    int i;
    for (i=0;i<m_nm;i++)
      h_c[i] += M.h_c[i];       
}

void Matrice::operator-=(const Matrice &M)
{
    int i;
    for (i=0;i<m_nm;i++)
      h_c[i] -= M.h_c[i];       
}

__global__ void multiplyGPU(double *w, double *u, double *v,
                            int n, int p, int m)
{
    int i, j, k;
    i = blockIdx.x*blockDim.x+threadIdx.x;
    j = blockIdx.y*blockDim.y+threadIdx.y;

    if (i >= n || j >= m) return;

    double s = 0.0;
    for (k=0; k<p; k++)
       s += u[k + i*p] * v[j + k*m];
    w[i+j*n] = s;
}

void multiply(Matrice & M1, const Matrice &M2, const Matrice &M3)
{
    int n = M2.n(), p = M2.m(), q = M3.n(), m = M3.m();
    if (p != q || n != M1.n() || m != M1.m())
        throw std::runtime_error("erreur de dimensions");


    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x)/blockSize.x,
                  (m + blockSize.y)/blockSize.y); 
          
    multiplyGPU<<<gridSize, blockSize>>>(M1.d_c, M2.d_c, M3.d_c, n, p, m);
    M1.m_synchronized = false;
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