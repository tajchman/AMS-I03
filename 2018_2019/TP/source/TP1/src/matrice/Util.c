

void affiche_(int *n, double *d)
{
  fprintf(stderr, "%5d : %15.7g\r");
}

void affiche(int n, double d)
{
  affiche_(&n, &d);
}
