#pragma once

#include <iostream>

class Matrice {
public:
    Matrice(int n, int m, const char *name);
    Matrice(const Matrice &other);
    void operator=(const Matrice &M);
    ~Matrice();

    void Identite();
    void Init(double v);
    void Random(double cmax);

    int n() const { return m_n; }
    int m() const { return m_m; }
    double & operator()(int i, int j) { return h_c[j + i*m_m]; }
    double operator()(int i, int j) const { return h_c[j + i*m_m]; }

    std::string & name() { return m_name; }
    std::string name() const { return m_name; }

    void operator-=(const Matrice &M);
    void operator+=(const Matrice &M);
    void operator*=(const Matrice &M);
    double norm2();

    bool synchronized() const { return m_synchronized; }
    void synchronized(bool s) { m_synchronized = s; }
    void synchronize();

    double * device() { return d_c; }
    double * host() { return h_c; }
    const double * device() const  { return d_c; }
    const double * host() const { return h_c; }

private:
    std::string m_name;
    double *h_c;
    double *d_c;
    int m_n, m_m, m_nm;
    size_t bytes;
    bool m_synchronized;

    void copyToDevice();
    void copyToHost();
};

void multiply(Matrice & M1, const Matrice &M2, const Matrice &M3);
std::ostream & operator<<(std::ostream & f, Matrice & M);
