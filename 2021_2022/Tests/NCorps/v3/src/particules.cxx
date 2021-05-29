#include "particules.hxx"
#include <cmath>

void Particules::move(double dt)
{
    const double softening = 1e-20;

    int nps = x.size();
    for (int i = 0; i < nps; i++) { 

      double Fx = 0.0, Fy = 0.0, Fz = 0.0; 
      
    for (int j = 0; j < nps; j++) { 
    
      const double dx = x[j] - x[i];
      const double dy = y[j] - y[i];
      const double dz = z[j] - z[i];
      const double dr  = 1.0/sqrt(dx*dx + dy*dy + dz*dz + softening);
      const double dr3 = dr*dr*dr;
	
      // Calculate the net force
      Fx += dx * dr3;  
      Fy += dy * dr3;  
      Fz += dz * dr3;

    }

    // Accelerate ps in response to the gravitational force
    vx[i] += dt*Fx; 
    vy[i] += dt*Fy; 
    vz[i] += dt*Fz;
  }

  // Move ps according to their velocities
  // O(N) work, so using a serial loop
  for (int i = 0 ; i < nps; i++) { 
    x[i]  += vx[i]*dt;
    y[i]  += vy[i]*dt;
    z[i]  += vz[i]*dt;
  }
}
