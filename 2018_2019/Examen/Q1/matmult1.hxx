void matmult1(std::vector<double> & W,
              const std::vector<double> & A,
              const std::vector<double> & V) {
  std::size_t i, j, n = W.size(), m = V.size();
  for (i=0; i<n; i++) {
    for (j=0; j<m; j++)
      W[i] += A[i*m+j] * V[j];
    }
}

