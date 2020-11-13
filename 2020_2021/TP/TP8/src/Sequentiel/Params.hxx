#ifndef __PARAMS_HXX__
#define __PARAMS_HXX__

#include <cstring>

struct sParams {
  sParams()
    : n(1000), n_it(100), it_output(0), n_th(1)
  {
  }

  sParams(int argc, char **argv)
    : n(1000), n_it(100), it_output(0), n_th(1)
  {
    for (int iarg = 1; iarg < argc; iarg++) {
      const char * s = argv[iarg];
      if (strncmp("n=", s, 2) == 0)
        n = strtol(s + 2, NULL, 10);
      else if (strncmp("it=", s, 3) == 0)
        n_it = strtol(s + 3, NULL, 10);
      else if (strncmp("out=", s, 4) == 0)
        it_output = strtol(s + 4, NULL, 10);
      else if (strncmp("threads=", s, 8) == 0)
        n_th = strtol(s + 8, NULL, 10);
      else if (strncmp("-h", s, 2) == 0)
        usage(argv[0], sParams());
    }
  }
  
  static void usage(const char *s, const sParams & p)
  {
    std::cerr << "\nusage " << s
              << " [n=<int>][it=<int>][out=<int>][threads=<int>]\n\n"
              << "parametres\n\n"
              << "\tn=<int>       taille du domaine (n x n), defaut n = "
              << p.n << "\n"
              << "\tit=<int>      nbre de pas de temps, defaut it = "
              << p.n_it << "\n"
              << "\tout=<int>     frequence de sortie des resultats, defaut out = "
              << p.it_output << "\n"
              << "\tthreads=<int> nombre de threads, defaut threads = "
              << p.n_th << "\n"
              << std::endl;
    exit (-1);
  }

  int n;
  int n_it;
  int it_output;
  int n_th;
};

#endif
