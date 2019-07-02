void matmult1(const std::vector<double> & A,
              const std::vector<double> & V,
              std::vector<double> & W)
{
  size_t i,j,n = W.size(), m = V.size();
  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
      W[i] += A[i*m + j] * V[j];
}
