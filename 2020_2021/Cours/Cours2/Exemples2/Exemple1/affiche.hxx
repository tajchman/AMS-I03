#include <iostream>
#include <vector>

inline void affiche(const char *message, const std::vector<double> & v)
{
  if (v.size() < 11) {
    std::cout << message;
    for (int i=0; i<v.size(); i++)
      std::cout << " " << v[i];
    std::cout << std::endl << std::endl;
  }
}

