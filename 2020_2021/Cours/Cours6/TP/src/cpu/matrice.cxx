#include "matrice.hxx"
#include "timer.hxx"
#include <cstring>
#include <cstdlib>
#include <stdexcept>

Matrice::Matrice(int n, int m, const char * name) 
    : m_n(n), m_m(m), m_nm(n*m), 
      m_c(nullptr),
      m_name(name)
{
    Timer T = GetTimer(0);
    T.start();
    m_c = new double[n*m];
    T.stop();
    std::cerr << T.elapsed() << std::endl;
}

Matrice::Matrice(const Matrice &other) 
    : m_n(other.m_n), 
      m_m(other.m_m), 
      m_nm(other.m_nm), 
      m_c(nullptr),
      m_name(other.m_name)
{
    Timer T0 = GetTimer(0);
    T0.start();
    m_c = new double[other.m_nm];
    T0.stop();

    Timer T1 = GetTimer(1);
    T1.start();
    std::memcpy(m_c, other.m_c, m_nm * sizeof(double));
    T1.stop();
}

void Matrice::operator=(const Matrice &other)
{
    if (&other == this)
        return;

    m_n = other.m_n;
    m_m = other.m_m;
    if (m_nm != other.m_nm) {
        Timer T = GetTimer(0);
        T.start();
        delete [] m_c;
        m_c = new double[other.m_nm];
        T.stop();
    }
    Timer T = GetTimer(1);
    T.start();
    m_n = other.m_n;
    m_m = other.m_m;
    std::memcpy(m_c, other.m_c, m_nm*sizeof(double));
    T.stop();
}

void Matrice::Init(double v) {
    int i;

    for (i=0; i<m_nm; i++)
        m_c[i] = v;
}

Matrice::~Matrice() {
    delete [] m_c;
}

void Matrice::Identite() {

    Init(0.0);
    for (int i=0; i<m_n;i++)
        (*this)(i,i) = 1.0;
}

void Matrice::Random(double cmax) {

    std::srand(0);
    int i,j;
    for (i=0; i<m_n;i++)
        for (j=0; j<m_m; j++) {
            double c = (cmax * std::rand())/RAND_MAX;
            (*this)(i,j) = c;
        }
    for (i=0; i<m_n;i++)
       (*this)(i,i) += 1.0;
}

inline double sqr(double x) { return x*x;}

double Matrice::norm2() {

    int i,j;
    double s = 0.0;
    for (i=0; i<m_n;i++)
        for (j=0; j<m_m; j++)
            s += sqr((*this)(i,j));
    return sqrt(s);
}

void Matrice::operator+=(const Matrice &M)
{
    int i;
    for (i=0;i<m_nm;i++)
      m_c[i] += M.m_c[i];       
}

void Matrice::operator-=(const Matrice &M)
{
    int i;
    for (i=0;i<m_nm;i++)
      m_c[i] -= M.m_c[i];       
}

void multiply(Matrice & M1, const Matrice &M2, const Matrice &M3)
{
    int i, j, k, n = M2.n(), p = M2.m(), q = M3.n(), m = M3.m();
    if (p != q || n != M1.n() || m != M1.m())
        throw std::runtime_error("erreur de dimensions");

    for (i=0; i<n; i++)
        for (j=0; j<m; j++) {
            double s = 0;
            for (k=0; k<p; k++)
                s += M2(i,k) * M3(k,j);
            M1(i,j) = s;
        }
}

#include <iomanip>

std::ostream & operator << (std::ostream & f, const Matrice & M)
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