#include "particules.hxx"
#include <cmath>

void Particules::move(reel dt)
{
    // Avoid singularity and interaction with self
    const reel softening = 1e-20;

    int nps = p.size();
    for (int i = 0; i < nps; i++) { 

      reel Fx = 0, Fy = 0, Fz = 0; 
      
    // Loop over ps that exert force: vectorization expected here
    for (int j = 0; j < nps; j++) { 
    
      // Newton's law of universal gravity
      const reel dx = p[j].x - p[i].x;
      const reel dy = p[j].y - p[i].y;
      const reel dz = p[j].z - p[i].z;
      const reel drSquared  = dx*dx + dy*dy + dz*dz + softening;
      const reel drPower32  = pow(drSquared, 3.0/2.0);
	
      // Calculate the net force
      Fx += dx / drPower32;  
      Fy += dy / drPower32;  
      Fz += dz / drPower32;

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
