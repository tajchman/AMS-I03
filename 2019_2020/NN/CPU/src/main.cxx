
#include "input.hxx"
#include "reseau.hxx"
#include <string>
#include <iostream>

int main(int argc, char ** argv)
{
  input In(argv[1], argv[2]);

  // nombre de couches de neurones
  std::vector<size_t> layer(3);

  // nombre de neurones dans chaque couche
  layer[0] = In.n();
  layer[1] = 16;
  layer[2] = 10;

  reseau NN(layer);

  vector p;
  double v, w;

  for (int k=0; k<100; k++) {
      In.next(p, v);
      w = NN.forward(p);
      std::cerr << "v = " << v << " w = " << w << std::endl;
  }
  return 0;
}
