#include <iostream>
#include "fmm_l2l.h"
#include "../vpmcore/common.h"
#include "fmm_utils.h"

__device__ __inline__ void static transZL2L(vpm::real* __restrict__ s_L, vpm::real* __restrict__ s_c, vpm::real z, int p)
{
	// m = 0
	for (int n = 0; n < p; ++n)
	{
		const int idx = (n * (n + 1));
		s_c[n] = s_L[idx];
		s_L[idx] = 0.0f;
	}

	vpm::real fac = 1.0f;
	for (int d = 0; d < p; ++d)
	{
		for (int n_ = d; n_ < p; ++n_)
		{
			const int n = n_ - d;
			s_L[(n * (n + 1))] += fac * s_c[n_];
		}
		fac *= -z / (d + 1);
	}

	// m > 0
	for (int m = 1; m < p; ++m)
	{
		// Cache column from s_L to s_c
		for (int n = m; n < p; ++n)
		{
			const int idxp = (n * (n + 1)) + m;
			const int idxm = (n * (n + 1)) - m;
			s_c[n - 1    ] = s_L[idxp];
			s_c[n - 1 + p] = s_L[idxm];
			s_L[idxp] = 0.0f;
			s_L[idxm] = 0.0f;
		}

		fac = 1.0f;
		for (int d = 0; d < p - m; ++d)
		{
			for (int n_ = d + m; n_ < p; ++n_)
			{
				const int n = n_ - d;
				s_L[(n * (n + 1)) + m] += fac * s_c[n_ - 1    ];
				s_L[(n * (n + 1)) - m] += fac * s_c[n_ - 1 + p];
			}
			fac *= -z / (d + 1);
		}
	}
}

__global__ void fmm::l2l(
	const vpm::nidx_t firstTarget,
	const vpm::nidx_t* __restrict__ sources,
	const vpm::vec3* __restrict__ distances,
	vpm::real* __restrict__ L,
	vpm::nidx_t count, int p
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
	vpm::real* const s_L = reinterpret_cast<vpm::real*>(indices + blockDim.x) + threadIdx.x * coefsExpansionPad;
	vpm::real* const s_r = reinterpret_cast<vpm::real*>(indices + blockDim.x) + blockDim.x * coefsExpansionPad + threadIdx.x * coefsCache;
	vpm::real* const s_Lw = reinterpret_cast<vpm::real*>(indices + blockDim.x) + warpIndex * warpSize * coefsExpansionPad;

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
				s_Lw[icell * coefsExpansionPad + icoef] = L[nodeIdx * coefsExpansion + icoef];
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
			s_Lw[icell * coefsExpansionPad + icoef] = L[nodeIdx * coefsExpansion + icoef];
			icoef += warpSize;
		}
	}

	__syncwarp();

	if (globalIndex < count)
	{
		const vpm::real x = distances[globalIndex].x;
		const vpm::real y = distances[globalIndex].y;
		const vpm::real z = distances[globalIndex].z;

		const vpm::real rxy2 = x * x + y * y;
		const vpm::real rxy = sqrt(rxy2);
		const vpm::real r = sqrt(rxy2 + z * z);
		const vpm::real cosAlpha = y / rxy;
		const vpm::real sinAlpha = -x / rxy;
		const vpm::real cosBeta = z / r;
		const vpm::real sinBeta = rxy / r;

		fmm::rotateZ(s_L, cosAlpha, sinAlpha, p);
		fmm::swapXZ<c_swapCoefsL>(s_L, s_r, p);
		fmm::rotateZ(s_L, cosBeta, sinBeta, p);
		fmm::swapXZ<c_swapCoefsL>(s_L, s_r, p);

		transZL2L(s_L, s_r, z, p);

		fmm::swapXZ<c_swapCoefsL>(s_L, s_r, p);
		fmm::rotateZ(s_L, cosBeta, -sinBeta, p);
		fmm::swapXZ<c_swapCoefsL>(s_L, s_r, p);
		fmm::rotateZ(s_L, cosAlpha, -sinAlpha, p);
	}

	__syncwarp();

	icell = 0;
	icoef = laneIndex;
	if (coefsExpansion < warpSize)
	{
		while (true)
		{
			if (icoef >= coefsExpansion)
			{
				icoef -= coefsExpansion;
				if (++icell >= warpNodesCount) break;
			}
			else
			{
				atomicAdd(&L[(firstTarget + icell) * coefsExpansion + icoef], s_Lw[icell * coefsExpansionPad + icoef]);
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
			}
			atomicAdd(&L[(firstTarget + icell) * coefsExpansion + icoef], s_Lw[icell * coefsExpansionPad + icoef]);
			icoef += warpSize;
		}
	}
}