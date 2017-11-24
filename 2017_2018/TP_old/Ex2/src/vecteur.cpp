#include "vecteur.hpp"

Vecteur::Vecteur(size_t n)
  : m_n(n), m_v(new double[n]) {
}

Vecteur::~Vecteur() {
  delete [] m_v;
}

