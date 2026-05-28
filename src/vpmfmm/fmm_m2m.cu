#include <iostream>
#include "fmm_m2m.h"
#include "../vpmcore/common.h"
#include "fmm_utils.h"

__device__ __inline__ void static transZM2M(vpm::real* __restrict__ s_M, vpm::real* __restrict__ s_c, vpm::real z, int p)
{
	for (int m = 1 - p; m < p; ++m)
	{
		const int absM = abs(m);

		// Cache column from s_M to s_c
		for (int n = absM; n < p; ++n)
		{
			const int idx = (n * (n + 1)) + m;
			s_c[n] = s_M[idx];
			s_M[idx] = 0.0f;
		}

		vpm::real fac = 1.0f;
		for (int d = 0; d < p - absM; ++d) // Consider changing loop order to trade extra reads/writes for redundant calculations of fac
		{
			for (int n = d + absM; n < p; ++n)
			{
				s_M[(n * (n + 1)) + m] += fac * s_c[n - d];
			}
			fac *= -z / (d + 1);
		}
	}
}

__global__ void fmm::m2m(
	const size_t* __restrict__ parents,
	const size_t* __restrict__ children,
	const vpm::vec3* __restrict__ distances,
	vpm::real* M,
	int p)
{
	size_t globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

	int firstChild = 100; // TODO: Get child offset
	size_t cellIndex = firstChild + globalIndex; 

	// Number of coefficients for each expansion
	const int coefsExpansion = p * p;
	// Number of coefficients in caching row
	const int coefsCache = 2 * p - 1;
	// Number of coefficients per thread, additional element for even p to avoid/reduce bank conflicts
	const int coefsThread = coefsExpansion + (p & 1 == 0);
	// Number of coefficients per warp
	const int coefsWarp = (coefsThread + coefsCache) * warpSize;

	// Current thread's lane and warp index
	const int laneId = threadIdx.x & (warpSize - 1);
	const int warpId = threadIdx.x >> 5;
	//const int globalWarpId = globalIndex >> 5;

	// Declare shared memory
	extern __shared__ vpm::real sh[];

	const int baseOffset = warpId * coefsWarp;
	const int globalBaseOffset = (firstChild + warpId * warpSize) * coefsExpansion;

	// This access patterns will cause occasional 1-degree bank conflicts for even p, TODO: improve??
	int icell = 0;
	int icoef = laneId;
	if (coefsExpansion < warpSize)
	{
		while (true)
		{
			while (icoef >= coefsExpansion)
			{
				icoef -= coefsExpansion;
				if (++icell >= warpSize) break;
			}
			sh[baseOffset + icell * coefsThread + icoef] = M[globalBaseOffset + icell * coefsExpansion + icoef];
			icoef += warpSize;
		}
	}
	else
	{
		while (true)
		{
			if (icoef >= coefsExpansion)
			{
				icoef -= coefsExpansion;
				if (++icell >= warpSize) break;
			}
			sh[baseOffset + icell * coefsThread + icoef] = M[globalBaseOffset + icell * coefsExpansion + icoef];
			icoef += warpSize;
		}
	}

	__syncwarp();

	// Shared memory contains all expansion coefficients first (with possible padding), then caching rows follow
	vpm::real* const s_M = sh + threadIdx.x * coefsThread;
	vpm::real* const s_r = sh + blockDim.x * coefsThread + threadIdx.x * coefsCache;

	const vpm::real x = distances[globalIndex].x;
	const vpm::real y = distances[globalIndex].y;
	const vpm::real z = distances[globalIndex].z;

	const vpm::real rxy2 = x * x + y * y;
	const vpm::real rxy = sqrtf(rxy2);
	const vpm::real r = sqrtf(rxy2 + z * z);

	// Rotate around z-axis by alpha = -atan2(x, y)
	fmm::rotateZ(s_M, y, -x, rxy, p);

	// Swap axes z <-> x
	fmm::swapXZ<c_swapCoefsM>(s_M, s_r, p);

	// Rotate around z-axis (originally x) by beta = atan2(rxy, z)
	fmm::rotateZ(s_M, z, rxy, r, p);

	// Swap axes z <-> x
	fmm::swapXZ<c_swapCoefsM>(s_M, s_r, p);

	// Translate along z axis
	transZM2M(s_M, s_r, z, p);

	// Swap axes z <-> x
	fmm::swapXZ<c_swapCoefsM>(s_M, s_r, p);

	// Rotate around z-axis (originally x) by beta = -atan2(rxy, z)
	fmm::rotateZ(s_M, z, -rxy, r, p);

	// Swap axes z <-> x
	fmm::swapXZ<c_swapCoefsM>(s_M, s_r, p);

	// Rotate around z-axis by alpha = atan2(x, y)
	fmm::rotateZ(s_M, y, -x, rxy, p);

	__syncwarp();

	// TODO: Improve simple copy back to global memory
	for (int i = 0; i < warpSize; ++i)
	{
		for (int j = 0; j < coefsExpansion; j += warpSize)
		{
			atomicAdd(&M[globalBaseOffset + i * coefsExpansion + j], sh[baseOffset + i * coefsThread + j]);
		}
	}

	return;
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

size_t M2MInfo::size() const
{
	size_t size = 0;
	for (const auto& entry : entries_) size += entry.parents_.size();
	return size;
}

void M2MInfo::add(int depth, size_t parent, size_t child, const vpm::vec3& distance)
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

void M2MInfo::addOffsets(const std::array<size_t, MAX_DEPTH>& depthOffsets)
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

void M2MInfo::toDevice() const
{
	if (dev_parents_ != nullptr) return;

	const size_t count = size();

	cudaMalloc((void**)&dev_parents_,	sizeof(size_t) * count);
	cudaMalloc((void**)&dev_children_,  sizeof(size_t) * count);
	cudaMalloc((void**)&dev_distances_, sizeof(vpm::vec3) * count);

	size_t offset = 0;
	std::vector<size_t> offsets;
	for (const auto& entry : entries_)
	{
		if (entry.size() == 0) continue;

		cudaMemcpy(dev_parents_   + offset, entry.parents_.data(),   sizeof(size_t)    * entry.parents_.size(),   cudaMemcpyHostToDevice);
		cudaMemcpy(dev_children_  + offset, entry.children_.data(),  sizeof(size_t)    * entry.children_.size(),  cudaMemcpyHostToDevice);
		cudaMemcpy(dev_distances_ + offset, entry.distances_.data(), sizeof(vpm::vec3) * entry.distances_.size(), cudaMemcpyHostToDevice);

		offset += entry.size();
	}
}

size_t* M2MInfo::dev_parents(int depth) const
{
	if (dev_parents_ == nullptr) return nullptr;

	size_t offset = 0;
	for (int i = 0; i < depth; ++i)
	{
		offset += entries_[i].size();
	}

	return dev_parents_ + offset;
}

size_t* M2MInfo::dev_children(int depth) const
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