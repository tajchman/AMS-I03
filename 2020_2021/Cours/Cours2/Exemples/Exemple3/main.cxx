#include <iostream>
#include <cmath>
#include <vector>

int main()
{
    int i;
    std::vector<double> u(10);
    double a;

    #pragma omp parallel for
    for (i=0; i<10; i++)
    {
        a = sin(i*0.01);
        u[i] = a*a;
    }

    std::cout << "u[5] = " << u[5] << std::endl;
    return 0;
}
