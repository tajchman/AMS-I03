inline void pause(int d) {

  volatile double v = 0;
  for(int n=0; n<d; ++n)
     for(int m=0; m<d; ++m)
         v += v*n*m;
};