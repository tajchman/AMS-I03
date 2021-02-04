void matmult_seq(Vecteur & W,
		 const Matrice & L,
		 const Vector & V)
{
  std::size_t i, j, n = L.size();

  for (i=0; i < n; i++)
    for (j=0; j <= i; j++)
      W(i) += L(i,j) * V(j)
}
