#ifdef __cplusplus
extern "C" {
#endif
  
  void waitKey();
  size_t memavail(double);

  void * start();
  double elapsed(void *t);
  void stop(void *);
  double time_precision();
		  
#ifdef __cplusplus
}
#endif
