#include "particules.hxx"
#include <cmath>

void Particules::move(double dt)
{
  const double softening = 1e-20;
  const int tileSize = 8;
  std::vector<double> Fx(tileSize), Fy(tileSize), Fz(tileSize);

  for (int i = 0; i < n; i+=tileSize) { 

#pragma omp simd
     for(int s=0; s<tileSize; s++)
     {
       Fx[s] = 0.0;
       Fy[s] = 0.0;
       Fz[s] = 0.0;
     }
      
    for (int j = 0; j < n; j++) { 
    
#pragma omp simd
      for (int s = 0; s < tileSize; s++) {
        const double dx = x[j] - x[i+s];
        const double dy = y[j] - y[i+s];
        const double dz = z[j] - z[i+s];
        const double dr  = 1.0/sqrt(dx*dx + dy*dy + dz*dz + softening);
        const double dr3 = dr*dr*dr;
	
        Fx[s] += dx * dr3;  
        Fy[s] += dy * dr3;  
        Fz[s] += dz * dr3;
      }
    }
      
#pragma omp simd
    for(int s=0; s<tileSize; s++)
    {
      vx[i+s] += dt*Fx[s]; 
      vy[i+s] += dt*Fy[s]; 
      vz[i+s] += dt*Fz[s];
    }
  }

  for (int i = 0 ; i < n; i++) { 
    x[i]  += vx[i]*dt;
    y[i]  += vy[i]*dt;
    z[i]  += vz[i]*dt;
  }
}
