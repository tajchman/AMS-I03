#include "particules.hxx"
#include <cmath>

void Particules::move(double dt)
{
    // Avoid singularity and interaction with self
    const double softening = 1e-20;

    int nps = p.size();
    for (int i = 0; i < nps; i++) { 

      double Fx = 0.0, Fy = 0.0, Fz = 0.0; 
      
    // Loop over ps that exert force: vectorization expected here
    for (int j = 0; j < nps; j++) { 
    
      // Newton's law of universal gravity
      const double dx = p[j].x - p[i].x;
      const double dy = p[j].y - p[i].y;
      const double dz = p[j].z - p[i].z;
      const double dr  = 1/sqrt(dx*dx + dy*dy + dz*dz + softening);
      const double dr3 = dr*dr*dr;
	
      // Calculate the net force
      Fx += dx * dr3;  
      Fy += dy * dr3;  
      Fz += dz * dr3;

    }

    // Accelerate ps in response to the gravitational force
    p[i].vx += dt*Fx; 
    p[i].vy += dt*Fy; 
    p[i].vz += dt*Fz;
  }

  // Move ps according to their velocities
  // O(N) work, so using a serial loop
  for (int i = 0 ; i < nps; i++) { 
    p[i].x  += p[i].vx*dt;
    p[i].y  += p[i].vy*dt;
    p[i].z  += p[i].vz*dt;
  }
}
