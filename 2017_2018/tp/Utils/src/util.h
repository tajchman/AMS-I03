#ifdef __cplusplus
extern "C" {
#endif
  
  void wait();
  size_t memavail(double);

  void * start();
  double elapsed(void *t);
  void stop(void *);
  double time_precision();
		  
#ifdef __cplusplus
}
#endif
