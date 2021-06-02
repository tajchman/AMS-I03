#include "particules.hxx"
#include <cmath>

void Particules::move(double dt)
{
  const double softening = 1e-20;

  __assume_aligned(vx, alignment);
  __assume_aligned(vy, alignment);
  __assume_aligned(vz, alignment);
  __assume_aligned(x,alignment);
  __assume_aligned(y,alignment);
  __assume_aligned(z,alignment);
 
 #pragma omp simd
  for (int i = 0; i < n; i++) { 

    double Fx = 0.0, 
           Fy = 0.0, 
           Fz = 0.0; 
    for (int j = 0; j < n; j++) { 
    
      const double dx = x[j] - x[i];
      const double dy = y[j] - y[i];
      const double dz = z[j] - z[i];
      const double dr  = sqrt(dx*dx + dy*dy + dz*dz + softening);
      const double dr3 = 1.0/(dr*dr*dr);
	
      Fy += dy * dr3;  
      Fz += dz * dr3;
      Fx += dx * dr3;  

    }

    vx[i] += dt*Fx; 
    vy[i] += dt*Fy; 
    vz[i] += dt*Fz;
  }

  for (int i = 0 ; i < n; i++) { 
    x[i]  += vx[i]*dt;
    y[i]  += vy[i]*dt;
    z[i]  += vz[i]*dt;
  }
  K = 0.0;
  for (int i = 0 ; i < n; i++) { 
   K += 0.5 * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
  }
}
