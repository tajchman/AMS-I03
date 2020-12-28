/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef __REDUCTION_H__
#define __REDUCTION_H__

double *work_reduce(int blocks);

double diff_reduce(size_t n,
	      const double *d_idata1,
	      const double *d_idata2,
	      double *d_work,
	      int threads,
	      int blocks);

#endif
