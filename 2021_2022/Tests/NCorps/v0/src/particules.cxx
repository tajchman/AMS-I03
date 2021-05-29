#include "particules.hxx"
#include <cmath>

void Particules::move(double dt)
{
    const double softening = 1e-20;

    int nps = p.size();
    for (int i = 0; i < nps; i++) { 

      double Fx = 0, Fy = 0, Fz = 0; 
      
    for (int j = 0; j < nps; j++) { 
    
      const double dx = p[j].x - p[i].x;
      const double dy = p[j].y - p[i].y;
      const double dz = p[j].z - p[i].z;
      const double drSquared  = dx*dx + dy*dy + dz*dz + softening;
      const double drPower32  = pow(drSquared, 3.0/2.0);
	
      // Calculate the net force
      Fx += dx / drPower32;  
      Fy += dy / drPower32;  
      Fz += dz / drPower32;

    }

    p[i].vx += dt*Fx; 
    p[i].vy += dt*Fy; 
    p[i].vz += dt*Fz;
  }

  for (int i = 0 ; i < nps; i++) { 
    p[i].x  += p[i].vx*dt;
    p[i].y  += p[i].vy*dt;
    p[i].z  += p[i].vz*dt;
  }
}
