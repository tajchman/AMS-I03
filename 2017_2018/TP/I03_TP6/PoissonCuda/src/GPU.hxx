/*
 * GPU.hxx
 *
 *  Created on: 4 févr. 2018
 *      Author: marc
 */

#ifndef GPU_HXX_
#define GPU_HXX_

#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		std::cerr << "CUDA call on line " << __LINE__                   \
		          << " returned error " << result << std::endl;         \
		exit(1);														\
	} }

#endif /* GPU_HXX_ */
