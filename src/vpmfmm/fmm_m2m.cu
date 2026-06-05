#include <iostream>
#include "fmm_m2m.h"
#include "../vpmcore/common.h"
#include "fmm_utils.h"

__device__ __inline__ void static transZM2M(vpm::real* __restrict__ s_M, vpm::real* __restrict__ s_c, vpm::real z, int p)
{
	// m = 0
	// Cache column from s_M to s_c
	for (int n = 0; n < p; ++n)
	{
		const int idx = (n * (n + 1));
		s_c[n] = s_M[idx];
		s_M[idx] = 0.0f;
	}

	vpm::real fac = 1.0f;
	for (int d = 0; d < p; ++d)
	{
		for (int n = d; n < p; ++n)
		{
			s_M[(n * (n + 1))] += fac * s_c[n - d];
		}
		fac *= -z / (d + 1);
	}

	// m > 0
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

		fac = 1.0f;
		for (int d = 0; d < p - m; ++d)
		{
			for (int n = d + m; n < p; ++n)
			{
				s_M[(n * (n + 1)) + m] += fac * s_c[n - d - 1    ];
				s_M[(n * (n + 1)) - m] += fac * s_c[n - d - 1 + p];
			}
			fac *= -z / (d + 1);
		}
	}
}

unsigned int fmm::shRequirementM2M(int p, int blockSize)
{
	const unsigned int coefsExpansion = blockSize * (p * p + ((p % 2) == 0)) * sizeof(vpm::real);
	const unsigned int coefsCache = blockSize * (2 * p - 1) * sizeof(vpm::real);
	const unsigned int nodeIndices = blockSize * sizeof(vpm::nidx_t);

	return coefsExpansion + coefsCache + nodeIndices;
}

__global__ void fmm::m2m(
	const vpm::nidx_t* __restrict__ targets,
	const vpm::nidx_t firstSource,
	const vpm::vec3* __restrict__ distances,
	vpm::real* __restrict__ M,
	vpm::nidx_t count, int p)
{
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
	vpm::real* const s_M = reinterpret_cast<vpm::real*>(indices + blockDim.x) + threadIdx.x * coefsExpansionPad;
	vpm::real* const s_r = reinterpret_cast<vpm::real*>(indices + blockDim.x) + blockDim.x * coefsExpansionPad + threadIdx.x * coefsCache;
	vpm::real* const s_Mw = reinterpret_cast<vpm::real*>(indices + blockDim.x) + warpIndex * warpSize * coefsExpansionPad;

	// This access patterns will cause occasional 1-degree bank conflicts for even p, TODO: improve??
	int icell = 0;
	int icoef = laneIndex;
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
				s_Mw[icell * coefsExpansionPad + icoef] = M[(firstSource + icell) * coefsExpansion + icoef];
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
			s_Mw[icell * coefsExpansionPad + icoef] = M[(firstSource + icell) * coefsExpansion + icoef];
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

		fmm::rotateZ(s_M, cosAlpha, sinAlpha, p);
		fmm::swapXZ<c_swapCoefsM>(s_M, s_r, p);
		fmm::rotateZ(s_M, cosBeta, sinBeta, p);
		fmm::swapXZ<c_swapCoefsM>(s_M, s_r, p);

		transZM2M(s_M, s_r, z, p);

		fmm::swapXZ<c_swapCoefsM>(s_M, s_r, p);
		fmm::rotateZ(s_M, cosBeta, -sinBeta, p);
		fmm::swapXZ<c_swapCoefsM>(s_M, s_r, p);
		fmm::rotateZ(s_M, cosAlpha, -sinAlpha, p);
	}

	// Load target indices into shared memory
	if (globalIndex < count)
		indices[threadIdx.x] = targets[globalIndex];

	__syncwarp();

	icell = 0;
	icoef = laneIndex;
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
				atomicAdd(&M[nodeIdx * coefsExpansion + icoef], s_Mw[icell * coefsExpansionPad + icoef]);
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
			atomicAdd(&M[nodeIdx * coefsExpansion + icoef], s_Mw[icell * coefsExpansionPad + icoef]);
			icoef += warpSize;
		}
	}
}

M2MInfo::~M2MInfo()
{
	if (dev_parents_ != nullptr)
	{
		cudaFree(dev_parents_);
		cudaFree(dev_children_);
		cudaFree(dev_distances_);
	}
}

vpm::nidx_t M2MInfo::size() const
{
	vpm::nidx_t size = 0;
	for (const auto& entry : entries_) size += entry.parents_.size();
	return size;
}

void M2MInfo::add(int depth, vpm::nidx_t parent, vpm::nidx_t child, const vpm::vec3& distance)
{
	if (depth >= entries_.size())
	{
		entries_.resize(depth + 1);
		std::cout << "Resized M2MInfo entries to " << entries_.size() << std::endl;
	}

	entries_[depth].parents_.push_back(parent);
	entries_[depth].children_.push_back(child);
	entries_[depth].distances_.push_back(distance);
}

void M2MInfo::addOffsets(const std::array<vpm::nidx_t, MAX_DEPTH>& depthOffsets)
{
	for (int depth = 1; depth < entries_.size(); ++depth)
	{
		for (int i = 0; i < entries_[depth].size(); ++i)
		{
			entries_[depth].children_[i] += depthOffsets[depth    ];
			entries_[depth].parents_[i]  += depthOffsets[depth - 1];
		}
	}
}

void M2MInfo::toDevice()
{
	if (dev_parents_ != nullptr) return;

	const vpm::nidx_t count = size();

	cudaMalloc((void**)&dev_parents_,	sizeof(vpm::nidx_t) * count);
	cudaMalloc((void**)&dev_children_,  sizeof(vpm::nidx_t) * count);
	cudaMalloc((void**)&dev_distances_, sizeof(vpm::vec3)   * count);

	vpm::nidx_t offset = 0;
	for (const auto& entry : entries_)
	{
		if (entry.size() == 0) continue;

		cudaMemcpy(dev_parents_   + offset, entry.parents_.data(),   sizeof(vpm::nidx_t) * entry.parents_.size(),   cudaMemcpyHostToDevice);
		cudaMemcpy(dev_children_  + offset, entry.children_.data(),  sizeof(vpm::nidx_t) * entry.children_.size(),  cudaMemcpyHostToDevice);
		cudaMemcpy(dev_distances_ + offset, entry.distances_.data(), sizeof(vpm::vec3)   * entry.distances_.size(), cudaMemcpyHostToDevice);

		offset += entry.size();
	}
}

vpm::nidx_t* M2MInfo::dev_parents(int depth) const
{
	if (dev_parents_ == nullptr) return nullptr;

	size_t offset = 0;
	for (int i = 0; i < depth; ++i)
	{
		offset += entries_[i].size();
	}

	return dev_parents_ + offset;
}

vpm::nidx_t* M2MInfo::dev_children(int depth) const
{
	if (dev_children_ == nullptr) return nullptr;

	size_t offset = 0;
	for (int i = 0; i < depth; ++i)
	{
		offset += entries_[i].size();
	}

	return dev_children_ + offset;
}

vpm::vec3* M2MInfo::dev_distances(int depth) const
{
	if (dev_distances_ == nullptr) return nullptr;

	size_t offset = 0;
	for (int i = 0; i < depth; ++i)
	{
		offset += entries_[i].size();
	}

	return dev_distances_ + offset;
}