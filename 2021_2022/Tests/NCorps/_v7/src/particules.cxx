#include "particules.hxx"
#include <cmath>

void Particules::move(double dt)
{
  const double softening = 1e-20;
  const int tileSize = 8;

#pragma omp parallel for
  for (int i = 0; i < n; i+=tileSize) { 

    double Fx[tileSize], Fy[tileSize], Fz[tileSize]; 
#pragma omp simd
     for(int s=0; s<tileSize; s++)
     {
       Fx[s] = 0.0;
       Fy[s] = 0.0;
       Fz[s] = 0.0;
     }
      
#pragma omp simd
    for (int j = 0; j < n; j++) { 
    
      for (int ii = i; ii < i + tileSize; ii++) {
      const double dx = x[j] - x[ii];
      const double dy = y[j] - y[ii];
      const double dz = z[j] - z[ii];
      const double dr  = 1.0/sqrt(dx*dx + dy*dy + dz*dz + softening);
      const double dr3 = dr*dr*dr;
	
      Fx[ii-i] += dx * dr3;  
      Fy[ii-i] += dy * dr3;  
      Fz[ii-i] += dz * dr3;
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
