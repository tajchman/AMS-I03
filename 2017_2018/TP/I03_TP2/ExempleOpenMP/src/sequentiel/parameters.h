#ifndef __PARAMETERS_H
#define __PARAMETERS_H

#ifdef __cplusplus
extern "C" {
#endif
  
  void * parseArgs(int argc, char **argv);

  double getDouble(void *p, const char * name, double Default);
  int getInt(void *p, const char * name, int Default);
  long getLong(void *p, const char * name, long Default);
  
  void setDouble(void *p, const char * name, double value);
  void setInt(void *p, const char * name, int value);
  void setLong(void *p, const char * name, long value);
  
  void freeArgs(void **p);
  
#ifdef __cplusplus
}
#endif
  
#endif
