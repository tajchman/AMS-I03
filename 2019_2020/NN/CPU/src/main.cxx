
#include "input.hxx"
#include "reseau.hxx"

int main(int argc, char ** argv)
{
  input In(argv[1], argv[2]);

  std::vector<size_t> layer(3);
  layer[0] = In.n();
  layer[1] = 16;
  layer[2] = 10;

  reseau NN(layer);

//  for (i_train = 0; i_train < 10; i_train++) {
//    std::cout << labelFile.next() << std::endl;
//    cImage I(ncols, nrows);
//    imageFile.next(I.coef);
//    std::string s("train_");
//    s += std::to_string(i_train);
//    s += ".png";
//
//    I.write_png(s.c_str());
//  }

  return 0;
}
