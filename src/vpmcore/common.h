#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <cuda.h>

#define ENABLE_CUDA_ERROR
#define TRANSPOSED
//#define DOUBLE_PRECISION
//#define CLASSIC_VPM
#define PINNED_MEMORY

#ifdef DOUBLE_PRECISION
	#define EPS 1e-9
	#define PI     3.14159265358979
	#define const1 0.06349363593424097
	#define const2 0.7978845608028654
	#define const3 0.238732414637843
	#define const4 0.07957747154594767
	#define sqrt2  1.4142135623730951
namespace vpm
{
	using real = double;
}
#else
	#define EPS 1e-6f
	#define PI     3.14159265358979f
	#define const1 0.06349363593424097f
	#define const2 0.7978845608028654f
	#define const3 0.238732414637843f
	#define const4 0.07957747154594767f
	#define sqrt2  1.4142135623730951f
namespace vpm
{
	using real = float;
}
#endif

namespace vpm
{
	using vec4 = glm::tvec4<real>;
	using vec3 = glm::tvec3<real>;
	using vec2 = glm::tvec2<real>;
	using mat3 = glm::tmat3x3<real>;
	using midx_t = uint64_t;	// type for octree node morton codes 
	using nidx_t = uint32_t;	// type for octree node indices
	using pidx_t = uint32_t;	// type for particle indices
}

static constexpr int MAX_DEPTH = 20;

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

struct CUDAKernelParams {
	int numBlocks;
	int blockSize;
	size_t sharedBytes;
	cudaStream_t stream;
};