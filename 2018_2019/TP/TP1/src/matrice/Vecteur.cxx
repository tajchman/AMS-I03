#include <cstring>
#include <cmath>
#include <ctime>
#include "Vecteur.hxx"

double Vecteur::normalise() {
  int i, n=size();

  double norme = 0.0;
  for (i=0; i<n; i++)
    norme += m_coeffs[i]*m_coeffs[i];
  norme = sqrt(norme);

  if (norme > 0)
    for (i=0; i<n; i++)
      m_coeffs[i] /= norme;

  return norme;
}
