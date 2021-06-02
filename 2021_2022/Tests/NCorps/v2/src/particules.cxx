#include "particules.hxx"
#include <cmath>

void Particules::move(float dt)
{
    const float softening = 1e-20f;

    int nps = p.size();
    for (int i = 0; i < nps; i++) { 

      float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f; 
      
    for (int j = 0; j < nps; j++) { 
    
      const float dx = p[j].x - p[i].x;
      const float dy = p[j].y - p[i].y;
      const float dz = p[j].z - p[i].z;
      const float dr  = 1.0f/sqrtf(dx*dx + dy*dy + dz*dz + softening);
      const float dr3 = dr*dr*dr;
	
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
