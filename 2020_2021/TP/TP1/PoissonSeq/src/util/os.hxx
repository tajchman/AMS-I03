#ifndef _OS_HXX
#define _OS_HXX

#include <sys/stat.h>

int mkdir_p(const char *path, const mode_t mode = S_IRWXU | S_IRGRP | S_IROTH);

#endif