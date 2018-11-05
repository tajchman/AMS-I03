#include <vector>
#include <cstring>
#include <cmath>
#include "Matrice.hpp"

int main(int argc, char **argv)
{
	int i, j, k, l, kmax, lmax;
	int n = argc > 1 ? strtol(argv[1], nullptr, 10) : 1000;
	int m = argc > 2 ? strtol(argv[2], nullptr, 10) : 2000;
	int p = argc > 3 ? strtol(argv[3], nullptr, 10) : 50;

	Matrice<double> a(n,m), b(m,n);

	const int blocksize = p;

	for (j = 0; j < m; j += blocksize) {
		lmax = j + blocksize; if (lmax > m) lmax = m;
		for ( i = 0; i < n; i += blocksize) {
			kmax = i + blocksize; if (kmax > n) kmax = n;
			// transpose the block beginning at [i,j]
			for (l = j; l < lmax; ++l)
				for (k = i; k < kmax; ++k)
					b(l,k) = a(k,l);
		}
	}

	return 0;
}
