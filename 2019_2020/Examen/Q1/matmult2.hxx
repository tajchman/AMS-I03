void matmult2(const std::vector<double> & A,
              const std::vector<double> & V,
              std::vector<double> & W)
{
  size_t i,j,n = W.size(), m = V.size();
  for (j=0; j<m; j++)
    for (i=0; i<n; i++)
      W[i] += A[i*m + j] * V[j];
}
