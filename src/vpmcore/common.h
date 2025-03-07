#pragma once

#include <glm/glm.hpp>

#define ENABLE_CUDA_ERROR
#define TRANSPOSED
//#define DOUBLE_PRECISION
//#define CLASSIC_VPM
#define PINNED_MEMORY
#define MEMCPY_ASYNC

#ifdef DOUBLE_PRECISION
	#define EPS 1e-9
	typedef glm::dvec3 vpmvec3;
	typedef glm::dvec2 vpmvec2;
	typedef glm::dmat3 vpmmat3;
	typedef double     vpmfloat;
	#define PI     3.14159265358979
	#define const1 0.06349363593424097
	#define const2 0.7978845608028654
	#define const3 0.238732414637843
	#define const4 0.07957747154594767
	#define sqrt2  1.4142135623730951
#else
	#define EPS 1e-6f
	typedef glm::fvec3 vpmvec3;
	typedef glm::fvec2 vpmvec2;
	typedef glm::fmat3 vpmmat3;
	typedef float      vpmfloat;
	#define PI     3.14159265358979f
	#define const1 0.06349363593424097f
	#define const2 0.7978845608028654f
	#define const3 0.238732414637843f
	#define const4 0.07957747154594767f
	#define sqrt2  1.4142135623730951f
#endif

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

inline void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#ifdef ENABLE_CUDA_ERROR
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
#endif
}