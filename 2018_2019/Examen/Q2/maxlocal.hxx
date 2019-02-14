void maxlocal(std::vector<double> & v,
	      std::vector<int> & imax,
	      std::size_t &nmax) {
  std::size_t i, n = v.size();
  std::size_t smax = imax.size();
  nmax = 0;
  for (i=1; i<n-1; i++)
    if ((v[i-1]<v[i]) && (v[i]>v[i+1])) {
      imax[nmax] = i;
      nmax += 1;
      if (nmax >= smax) break;
    }
}
