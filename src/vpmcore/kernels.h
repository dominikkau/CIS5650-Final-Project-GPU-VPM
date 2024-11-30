#pragma once

#include <glm/glm.hpp>

// Constants
constexpr float PI = 3.14159265358979f;
constexpr float const1 = powf(1.0f / (2.0f * PI), 1.5f);
constexpr float const2 = sqrtf(2.0f / PI);
constexpr float const3 = 3.0f / (4.0f * PI);
constexpr float const4 = 1.0f / (4.0f * PI);
constexpr float sqrt2 = sqrtf(2.0f);

// TODO: check if g_dgdr is necessary (compute is free)

class SingularKernel {
public:
	__host__ __device__ inline float zeta(float r) { return (r == 0.0f) ? 1.0f : 0.0f; }
	__host__ __device__ inline float g   (float r) { return 1.0f; }
	__host__ __device__ inline float dgdr(float r) { return 0.0f; }
	__host__ __device__ inline glm::vec2 g_dgdr(float r) { return glm::vec2{ 1.0f, 0.0f }; }
}

class GaussianKernel {
public:
	__host__ __device__ inline float zeta(float r) { 
		return const3 * expf(- r*r*r);
	}
	__host__ __device__ inline float g   (float r) { 
		return 1.0f - expf(- r*r*r);
	}
	__host__ __device__ inline float dgdr(float r) { 
		return 3.0f * r*r * expf(- r*r*r);
	}
	__host__ __device__ inline glm::vec2 g_dgdr(float r) {
		float tmp = expf(- r*r*r);
		return glm::vec2{ 1.0f - tmp, 3.0f * r*r * tmp };
	}
}

class GaussianErfKernel {
public:
	__host__ __device__ inline float zeta(float r) {
		return const1 * expf(- r*r / 2.0f);
	}
	__host__ __device__ inline float g   (float r) {
		return erff(r / sqrt2) - const2 * r * expf(- r*r / 2.0f);
	}
	__host__ __device__ inline float dgdr(float r) { 
		return const2 * r*r * expf(- r*r / 2.0f);
	}
	__host__ __device__ inline glm::vec2 g_dgdr(float r) {
		float tmp = const2 * r * expf(- r*r / 2.0f);
		return glm::vec2{ erff(r / sqrt2) - tmp, r * tmp };
	}
}

class WinckelmansKernel {
public:
	__host__ __device__ inline float zeta(float r) {
		return const4 * 7.5f / powf(r*r + 1.0f, 3.5f);
	}
	__host__ __device__ inline float g(float r) {
		return r*r*r * (r*r + 2.5f) / powf(r*r + 1.0f, 2.5f)
	}
	__host__ __device__ inline float dgdr(float r) {
		return 7.5f * r*r / powf(r*r + 1.0f, 3.5f);
	}
	__host__ __device__ inline glm::vec2 g_dgdr(float r) {
		float tmp =  powf(r*r + 1.0f, 2.5f)
		return glm::vec2{ r*r*r * (r*r + 2.5f) / tmp,
					      7.5f * r*r / (tmp * (r*r + 1.0f)) };
	}
}