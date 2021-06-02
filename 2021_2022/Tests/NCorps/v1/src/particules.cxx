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
      const double dr  = sqrt(dx*dx + dy*dy + dz*dz + softening);
      const double dr3 = 1.0/(dr*dr*dr);
	
      Fx += dx * dr3;  
      Fy += dy * dr3;  
      Fz += dz * dr3;

    }

    p[i].vx += dt*Fx; 
    p[i].vy += dt*Fy; 
    p[i].vz += dt*Fz;
  }

  K = 0.0;
  for (int i = 0 ; i < nps; i++) { 
    p[i].x  += p[i].vx*dt;
    p[i].y  += p[i].vy*dt;
    p[i].z  += p[i].vz*dt;
    K += 0.5 * (p[i].vx*p[i].vx + p[i].vy*p[i].vy + p[i].vz*p[i].vz);
  }
}
