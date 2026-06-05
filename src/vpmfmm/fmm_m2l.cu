#include <iostream>
#include "fmm_m2l.h"
#include "../vpmcore/common.h"
#include "fmm_utils.h"

__device__ __inline__ void static transZM2L(vpm::real* __restrict__ s_M, vpm::real* __restrict__ s_c, vpm::real z, int p)
{
	const vpm::real invz = 1.0f / z;
	vpm::real fac0 = invz;
	vpm::real fac = fac0;

	// Cache column from s_M to s_c
	for (int n = 0; n < p; ++n)
	{
		const int idx = (n * (n + 1));
		s_c[n] = s_M[idx];
		s_M[idx] = 0.0f;
	}
	for (int s = 0; s < 2 * p - 1; ++s)
	{
		for (int n = max(0, s - p + 1); n < min(p, s + 1); ++n)
		{
			s_M[(n * (n + 1))] += fac * s_c[s - n];
		}
		fac *= (s + 1.0f) * invz;
	}

	fac0 *= -2.0f * invz * invz;
	for (int m = 1; m < p; ++m)
	{
		// Cache column from s_M to s_c
		for (int n = m; n < p; ++n)
		{
			const int idxp = (n * (n + 1)) + m;
			const int idxm = (n * (n + 1)) - m;
			s_c[n - 1    ] = s_M[idxp];
			s_c[n - 1 + p] = s_M[idxm];
			s_M[idxp] = 0.0f;
			s_M[idxm] = 0.0f;
		}

		vpm::real fac = fac0;
		for (int s = 2 * m; s < 2 * p - 1; ++s)
		{
			for (int n = max(m, s - p + 1); n < min(p, s - m + 1); ++n)
			{
				s_M[(n * (n + 1)) + m] += fac * s_c[s - n - 1];
				s_M[(n * (n + 1)) - m] += fac * s_c[s - n - 1 + p];
			}
			fac *= (s + 1.0f) * invz;
		}

		fac *= -(2.0f * m + 1.0f) * (2.0f * m + 2.0f) * invz * invz;
	}
}

__global__ void fmm::m2l(
	const vpm::nidx_t* __restrict__ targets,
	const vpm::nidx_t* __restrict__ sources,
	const vpm::vec3* __restrict__ distances,
	const vpm::real* __restrict__ M,
	vpm::real* __restrict__ L,
	vpm::nidx_t count, int p, bool reversed
){
	// Global index of this thread
	const size_t globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	// Index of warp in this block
	const int warpIndex = threadIdx.x >> 5;
	// Index within this warp
	const int laneIndex = threadIdx.x & (warpSize - 1);
	// First global index processed by this warp
	const int firstWarpIdx = blockIdx.x * blockDim.x + warpIndex * warpSize;
	
	// Compute number of nodes processed by this warp (32 or less for the last warp)
	const int warpNodesCount = firstWarpIdx < count ? min(warpSize, count - firstWarpIdx) : 0;
	// Return if entire warp is out of bounds
	if (warpNodesCount == 0) return;

	// Number of coefficients for each expansion
	const int coefsExpansion = p * p;
	// Number of coefficients for each expansion (including possible padding)
	const int coefsExpansionPad = p * p + ((p & 1) == 0);
	// Number of coefficients in caching row
	const int coefsCache = 2 * p - 1;

	// Declare shared memory and shared memory pointers
	extern __shared__ int sh[];
	vpm::nidx_t* const indices = reinterpret_cast<vpm::nidx_t*>(sh);
	vpm::real* const s_M  = reinterpret_cast<vpm::real*>(indices + blockDim.x) + threadIdx.x * coefsExpansionPad;
	vpm::real* const s_r  = reinterpret_cast<vpm::real*>(indices + blockDim.x) + blockDim.x * coefsExpansionPad + threadIdx.x * coefsCache;
	vpm::real* const s_Mw = reinterpret_cast<vpm::real*>(indices + blockDim.x) + warpIndex * warpSize * coefsExpansionPad;

	// Load source indices into shared memory
	if (globalIndex < count)
		indices[threadIdx.x] = sources[globalIndex];

	__syncwarp();

	// This access patterns will cause occasional 1-degree bank conflicts for even p, TODO: improve??
	int icell = 0;
	int icoef = laneIndex;
	int nodeIdx = indices[warpIndex * warpSize];
	if (coefsExpansion < warpSize)
	{
		while (true)
		{
			if (icoef >= coefsExpansion)
			{
				icoef -= coefsExpansion;
				if (++icell >= warpNodesCount) break;
				nodeIdx = indices[warpIndex * warpSize + icell];				
			}
			else
			{
				s_Mw[icell * coefsExpansionPad + icoef] = M[nodeIdx * coefsExpansion + icoef];
				icoef += warpSize;
			}
		}
	}
	else
	{
		while (true)
		{
			if (icoef >= coefsExpansion)
			{
				icoef -= coefsExpansion;
				if (++icell >= warpNodesCount) break;
				nodeIdx = indices[warpIndex * warpSize + icell];
			}
			s_Mw[icell * coefsExpansionPad + icoef] = M[nodeIdx * coefsExpansion + icoef];
			icoef += warpSize;
		}
	}

	__syncwarp();

	if (globalIndex < count)
	{
		const vpm::real x = reversed ? -distances[globalIndex].x : distances[globalIndex].x;
		const vpm::real y = reversed ? -distances[globalIndex].y : distances[globalIndex].y;
		const vpm::real z = reversed ? -distances[globalIndex].z : distances[globalIndex].z;

		const vpm::real rxy2 = x * x + y * y;
		const vpm::real rxy = sqrt(rxy2);
		const vpm::real r = sqrt(rxy2 + z * z);
		const vpm::real cosAlpha = y / rxy;
		const vpm::real sinAlpha = -x / rxy;
		const vpm::real cosBeta = z / r;
		const vpm::real sinBeta = rxy / r;

		fmm::rotateZ(s_M, cosAlpha, sinAlpha, p);
		fmm::swapXZ<c_swapCoefsM>(s_M, s_r, p);
		fmm::rotateZ(s_M, cosBeta, sinBeta, p);
		fmm::swapXZ<c_swapCoefsM>(s_M, s_r, p);

		transZM2L(s_M, s_r, z, p);

		fmm::swapXZ<c_swapCoefsL>(s_M, s_r, p);
		fmm::rotateZ(s_M, cosBeta, -sinBeta, p);
		fmm::swapXZ<c_swapCoefsL>(s_M, s_r, p);
		fmm::rotateZ(s_M, cosAlpha, -sinAlpha, p);

		// Load target indices into shared memory
		indices[threadIdx.x] = targets[globalIndex];
	}

	__syncwarp();

	icell = 0;
	icoef = laneIndex;
	nodeIdx = indices[warpIndex * warpSize];
	if (coefsExpansion < warpSize)
	{
		while (true)
		{
			if (icoef >= coefsExpansion)
			{
				icoef -= coefsExpansion;
				if (++icell >= warpNodesCount) break;
				nodeIdx = indices[warpIndex * warpSize + icell];
			}
			else
			{
				atomicAdd(&L[nodeIdx * coefsExpansion + icoef], s_Mw[icell * coefsExpansionPad + icoef]);
				icoef += warpSize;
			}
		}
	}
	else
	{
		while (true)
		{
			if (icoef >= coefsExpansion)
			{
				icoef -= coefsExpansion;
				if (++icell >= warpNodesCount) break;
				nodeIdx = indices[warpIndex * warpSize + icell];
			}
			atomicAdd(&L[nodeIdx * coefsExpansion + icoef], s_Mw[icell * coefsExpansionPad + icoef]);
			icoef += warpSize;
		}
	}
}

M2LInfo::~M2LInfo()
{
	if (dev_targets_ != nullptr)
	{
		cudaFree(dev_targets_);
		cudaFree(dev_sources_);
		cudaFree(dev_distances_);
	}
}

void M2LInfo::add(vpm::nidx_t target, vpm::nidx_t source, const vpm::vec3& distance)
{
	targets_.push_back(target);
	sources_.push_back(source);
	distances_.push_back(distance);
}

void M2LInfo::toDevice() const
{
	if (dev_targets_ != nullptr) return;

	const size_t count = targets_.size();

	cudaMalloc((void**)&dev_targets_, sizeof(vpm::nidx_t) * count);
	cudaMalloc((void**)&dev_sources_, sizeof(vpm::nidx_t) * count);
	cudaMalloc((void**)&dev_distances_, sizeof(vpm::vec3) * count);

	cudaMemcpy(dev_targets_,   targets_.data(),   sizeof(vpm::nidx_t) * count, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sources_,   sources_.data(),   sizeof(vpm::nidx_t) * count, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_distances_, distances_.data(), sizeof(vpm::vec3) * count, cudaMemcpyHostToDevice);
}