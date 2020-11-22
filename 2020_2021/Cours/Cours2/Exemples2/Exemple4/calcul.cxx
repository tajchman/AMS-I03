#include "calcul.hxx"
#include "timer.hxx"
#include <iostream>
#include <thread>
#include <cmath>

// somme des composantes d'un vecteur sur un intervalle
void somme(const Donnees &v, 
           int i0, int i1, double &s)
{
  int i;
  s = 0;
  for (i=i0; i<i1; i++) 
    s += sin(v[i].s);
}

void somme_modifie(const Donnees &v, 
           int i0, int i1, double &s)
{
  int i;
  double t = 0;
  for (i=i0; i<i1; i++) 
    t += sin(v[i].s);
  
  s = t;
}

double somme0(const Donnees & v)
{
  Timer T;
  T.start();

  double s;
  somme(v, 0, v.size(), s);

  T.stop();
  std::cout << "Temps CPU : " << T.elapsed() << std::endl;

  return s;
}

double somme1(const Donnees & v, int nThreads)
{
  Timer T;
  T.start();

  std::vector<std::thread> vth;
  int iTh;
  int i0, i1, n = v.size(), di = (n + nThreads)/nThreads;
  std::vector<double> s(nThreads);

   for (iTh = 0, i0 = 0; iTh<nThreads; iTh++, i0 += di)
  {
    i1 = i0 + di; if (i1 > n) i1 = n;
    vth.push_back(std::thread(somme, std::cref(v), i0, i1, std::ref(s[iTh])));
  }
  
  double s_total = 0;
  for (iTh = 0; iTh<nThreads; iTh++) {
    vth[iTh].join();
    s_total += s[iTh];
  }
  
  T.stop();
  std::cout << "Temps CPU : " << T.elapsed() << std::endl;
  
  return s_total;
}

double somme2(const Donnees & v, int nThreads, int offset)
{
  Timer T;
  T.start();

  std::vector<std::thread> vth;
  int iTh;
  int i0, i1, n = v.size(), di = (n + nThreads)/nThreads;
  std::vector<double> s(nThreads*offset);

  for (iTh = 0, i0 = 0; iTh<nThreads; iTh++, i0 += di)
  {
    i1 = i0 + di; if (i1 > n) i1 = n;
    vth.push_back(std::thread(somme, std::cref(v), i0, i1, std::ref(s[iTh*offset])));
  }

  double s_total = 0;
  for (iTh = 0; iTh<nThreads; iTh++) {
    vth[iTh].join();
    s_total += s[iTh*offset];
  }
  
  T.stop();
  std::cout << "Temps CPU : " << T.elapsed() << std::endl;

  return s_total;
}


double somme3(const Donnees & v, int nThreads)
{
  Timer T;
  T.start();

  std::vector<std::thread> vth;
  int iTh;
  int i0, i1, n = v.size(), di = (n + nThreads)/nThreads;
  std::vector<double> s(nThreads);

   for (iTh = 0, i0 = 0; iTh<nThreads; iTh++, i0 += di)
  {
    i1 = i0 + di; if (i1 > n) i1 = n;
    vth.push_back(std::thread(somme_modifie, std::cref(v), i0, i1, std::ref(s[iTh])));
  }
  
  double s_total = 0;
  for (iTh = 0; iTh<nThreads; iTh++) {
    vth[iTh].join();
    s_total += s[iTh];
  }
  
  T.stop();
  std::cout << "Temps CPU : " << T.elapsed() << std::endl;
  
  return s_total;
}

