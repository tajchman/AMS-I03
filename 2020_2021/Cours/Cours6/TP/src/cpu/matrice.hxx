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

private:
    std::string m_name;
    double *h_c;
    int m_n, m_m, m_nm;
};

void multiply(Matrice & M1, const Matrice &M2, const Matrice &M3);
std::ostream & operator<<(std::ostream & f, Matrice & M);
