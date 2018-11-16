#include <stdio.h>

void affiche_(int *n, double *d)
{
  fprintf(stderr, "%5d : %22.16g\r", *n, *d);
}

void affiche(int n, double d)
{
  affiche_(&n, &d);
}
