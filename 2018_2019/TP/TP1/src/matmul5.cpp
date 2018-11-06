/*
 * matmul5.cpp
 *
 *  Created on: 5 nov. 2018
 *      Author: marc
 */


#include <cstring>
#include <cmath>
#include "MatriceBloc.hpp"

int main(int argc, char **argv)
{
	int i, j, k, l, kmax, lmax;
	int n = argc > 1 ? strtol(argv[1], nullptr, 10) : 1000;
	int m = argc > 2 ? strtol(argv[2], nullptr, 10) : 2000;
	int p = argc > 3 ? strtol(argv[3], nullptr, 10) : 50;

	MatriceBloc a(n,m,p,p), b(m,n,p,p);
    init(a);

	transpose(b, a);

	if (n<10 && m<10) {
	     std::cout << "A" << std::endl << a << std::endl;
	     std::cout << "B" << std::endl << b << std::endl;
	}
	return 0;
}


