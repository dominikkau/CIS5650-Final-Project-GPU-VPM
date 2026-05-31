#include <cuda.h>
#include <iostream>
#include <vector>
#include "../vortexringsimulation.hpp"
#include "fmm_p2m.h"
#include <device_launch_parameters.h>
//#include <cooperative_groups.h>
//#include <cooperative_groups/reduce.h>
//#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
#define __CUDACC__
#endif // __INTELLISENSE__

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__

namespace cg = cooperative_groups;

template <typename AccumulatorFn>
__host__ __device__ __forceinline__ void rbsf(vpm::vec3 pos, vpm::real q, int p, AccumulatorFn acc)
{
	const vpm::real x = pos.x;
	const vpm::real y = pos.y;
	const vpm::real z = pos.z;
    const vpm::real r2 = x * x + y * y + z * z;

    // Define initial values for recurrence
    vpm::real Rnm1mm1pos =             q; // R_{n-1}^{m-1}
    vpm::real Rnmneg     = -0.5f * y * q; // R_{n}^{-m}
    vpm::real Rnmm1pos   =         z * q; // R_{n}^{m-1}
    vpm::real Rnmpos     =  0.5f * x * q; // R_{n}^{m}

    acc(0, Rnm1mm1pos);
    acc(1, Rnmneg);
    acc(2, Rnmm1pos);
    acc(3, Rnmpos);

    vpm::real Rnm1mm1neg; // R_{n-1}^{-(m-1)}
    vpm::real Rnmm1neg;   // R_{n}^{-(m-1)}  

    for (int n_ = 2; n_ < p; n_++)
    {
        const int n_2 = n_ * n_;
        const vpm::real facz = z * (2.0f * n_ - 1.0f);
        const vpm::real Rnp1mm1pos = (facz * Rnmm1pos - r2 * Rnm1mm1pos) / n_2;
        acc(n_2 + n_, Rnp1mm1pos);
        Rnm1mm1pos = Rnmm1pos;
        Rnmm1pos = Rnp1mm1pos;
    }

    Rnm1mm1neg = Rnmneg;
    Rnm1mm1pos = Rnmpos;

    for (int n = 2; n < p; n++)
    {
        int n2 = n * n;
        vpm::real div = 0.5f / n;
        Rnmneg = div * (x * Rnm1mm1neg - y * Rnm1mm1pos);
        Rnmpos = div * (x * Rnm1mm1pos + y * Rnm1mm1neg);

        acc(n2, Rnmneg);
        acc(n2 + 2 * n, Rnmpos);

        Rnmm1neg = z * Rnm1mm1neg;
        Rnmm1pos = z * Rnm1mm1pos;

        acc(n2 + 1, Rnmm1neg);
        acc(n2 + 2 * n - 1, Rnmm1pos);

        const int m = n - 1;
        const int m2 = m * m;
        for (int n_ = n + 1; n_ < p; n_++)
        {
            n2 = n_ * n_;
            div = 1.0f / (n2 - m2);
            const vpm::real facz = z * (2 * n_ - 1);
            const vpm::real Rnp1mm1neg = (facz * Rnmm1neg - r2 * Rnm1mm1neg) * div;
            const vpm::real Rnp1mm1pos = (facz * Rnmm1pos - r2 * Rnm1mm1pos) * div;

            acc(n2 + n_ - m, Rnp1mm1neg);
            acc(n2 + n_ + m, Rnp1mm1pos);

            Rnm1mm1neg = Rnmm1neg;
            Rnm1mm1pos = Rnmm1pos;
            Rnmm1neg = Rnp1mm1neg;
            Rnmm1pos = Rnp1mm1pos;
        }

        Rnm1mm1neg = Rnmneg;
        Rnm1mm1pos = Rnmpos;
    }
}

__device__ __forceinline__ void evalRbsfDirect(vpm::vec3 pos, vpm::real q, int p, vpm::real* M)
{
	auto acc = [=](int idx, vpm::real value)
	    {
		    M[idx] += value;
	    };
	rbsf(pos, q, p, acc);
}

__device__ __forceinline__ void evalRbsfDirectAtomic(vpm::vec3 pos, vpm::real q, int p, vpm::real* M)
{
	auto acc = [=](int idx, vpm::real value)
	    {
		    atomicAdd(&M[idx], value);
	    };
	rbsf(pos, q, p, acc);
}

template <typename Group>
__device__ __forceinline__ void evalRbsfReduce(vpm::vec3 pos, vpm::real q, int p, vpm::real* M, Group& group, bool done)
{
    auto acc = [=](int idx, vpm::real value)
        {
            const auto sum = cg::reduce(group, value, cg::plus<vpm::real>());
            if (group.thread_rank() == 0) M[idx] += sum;
        };
    rbsf(pos, q, p, acc);
}

template <typename Group>
__device__ __forceinline__ void evalRbsfReduceAtomic(vpm::vec3 pos, vpm::real q, int p, vpm::real* M, Group& group)
{
	auto acc = [=](int idx, vpm::real value)
	    {
		    const auto sum = cg::reduce(group, value, cg::plus<vpm::real>());
		    if (group.thread_rank() == 0) atomicAdd(&M[idx], sum);
	    };
	rbsf(pos, q, p, acc);
}

__device__ __forceinline__ int getMask(int it, vpm::pidx_t ip, vpm::pidx_t pmin, vpm::pidx_t pmax, int& numThreads, int& leader)
{
    vpm::pidx_t nl = min(ip - pmin, it);
    vpm::pidx_t nu = min(pmax - ip - 1, 32 - it - 1);

    int mask = 1 << it;
    mask |= ((1 << nl) - 1) << (it - nl);
    mask |= ((1 << nu) - 1) << (it + 1);

    numThreads = nl + nu + 1;
    leader = it - nl;

    return mask;
}

__device__ __forceinline__ void evalRbsfReduceManual(vpm::vec3 pos, vpm::real q, int p, vpm::real* M, int ip, int pmin, int pmax, int it)
{
    int mask, numThreads, leader;
    mask = getMask(it, ip, pmin, pmax, numThreads, leader);
	int roundedWidth = numThreads <= 1 ? 1 : 1 << (32 - __clz(numThreads - 1));

    auto acc = [=](int idx, vpm::real value)
        {
            vpm::real tmp = 0.0f;

            #pragma unroll
            for (int i = 32; i > 0; i >>= 1)
            {
                tmp = __shfl_down_sync(mask, value, i);
                value += (it + i < leader + numThreads) ? tmp : 0.0f;
            }
            if (it == leader) M[idx] += value;
        };
    rbsf(pos, q, p, acc);
}

unsigned int fmm::shRequirementP2M(int p, int blockSize)
{
    int threadsPerNode = 2;
    blockSize /= threadsPerNode;
    const unsigned int coefsExpansion = blockSize * (p * p + ((p % 2) == 0)) * sizeof(vpm::real);
    const unsigned int pointIndices = (blockSize + 1) * sizeof(vpm::pidx_t);
	const unsigned int centers = blockSize * sizeof(vpm::vec3);

	return coefsExpansion + pointIndices + centers;
}

// Computes Multipole expansion of particles
// Evaluates regular spherical basis function
__global__ void fmm::p2m(const vpm::nidx_t* __restrict__ nodes, const vpm::pidx_t* __restrict__ pointsEnd, 
    const vpm::vec3* __restrict__ centers, size_t count, const vpm::vec3* __restrict__ xs, 
    const vpm::real* __restrict__ qs, vpm::real* M, int p)
{
    const size_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int threadsPerNode = 2;
	//constexpr int logThreadsPerNode = 1;
	constexpr int warpSize = 32;
	constexpr int nodesPerWarp = warpSize / threadsPerNode;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    // If entire warp is inactive, we can return immediately
    const int firstIndexWarp = blockIdx.x * blockDim.x / 2 + warp.meta_group_rank() * nodesPerWarp;
    if (firstIndexWarp >= count) return;

    // Active threads
    const int effectiveNodes = count - firstIndexWarp > nodesPerWarp ? nodesPerWarp : count - firstIndexWarp;

	const int coefsExpansion = p * p + 1 - (p & 1);

    const int firstIndex = block.group_index().x * block.dim_threads().x / threadsPerNode;


    extern __shared__ int sh[];
	// Calculate shared memory offsets on a block level
    vpm::pidx_t* s_ptIdxs = reinterpret_cast<vpm::pidx_t*>(sh);
	vpm::vec3* s_Centers = reinterpret_cast<vpm::vec3*>(s_ptIdxs + block.dim_threads().x / threadsPerNode + 1);
	vpm::real* s_M = reinterpret_cast<vpm::real*>(s_Centers + block.dim_threads().x / threadsPerNode);
    // Calculate shared memory offsets for each warp
    s_ptIdxs += warp.meta_group_rank() * nodesPerWarp;
	s_Centers += warp.meta_group_rank() * nodesPerWarp;
	s_M += warp.meta_group_rank() * coefsExpansion * nodesPerWarp;

    // Load point indices into shared memory
    if (block.thread_rank() == 0)
    {
		if (block.group_index().x == 0) s_ptIdxs[0] = 0;
		else                            s_ptIdxs[0] = pointsEnd[firstIndex - 1];
    }
    if ((globalIdx/2 < count) && (warp.thread_rank() & 1))
    {
        s_ptIdxs[warp.thread_rank() / threadsPerNode + 1] = pointsEnd[globalIdx / threadsPerNode];
        s_Centers[warp.thread_rank() / threadsPerNode] = centers[globalIdx / threadsPerNode];
    }

    for (int i = warp.thread_rank(); i < coefsExpansion * effectiveNodes; i += 32)
    {
        s_M[i] = 0.0f;
    }
    block.sync();

	vpm::pidx_t ptsIdx = s_ptIdxs[0] + warp.thread_rank();
    vpm::pidx_t startIdx = s_ptIdxs[0];
    vpm::pidx_t endIdx = s_ptIdxs[1];
    vpm::nidx_t cellIdx = 0;
	vpm::vec3 center = s_Centers[0];
	const vpm::pidx_t warpEndIdx = s_ptIdxs[effectiveNodes];
    while ((ptsIdx >= endIdx) && (ptsIdx < warpEndIdx))
    {
		if (++cellIdx >= effectiveNodes) break;
        startIdx = endIdx;
        endIdx = s_ptIdxs[cellIdx + 1];
        center = s_Centers[cellIdx];
    }
    while (ptsIdx < warpEndIdx) //TODO: reformulate as for loop
    {
        warp.sync();
        vpm::real q = qs[ptsIdx];
        vpm::vec3 x = xs[ptsIdx] - center;

		evalRbsfReduceManual(x, q, p, s_M + coefsExpansion * cellIdx, ptsIdx, startIdx, endIdx, warp.thread_rank());

		ptsIdx += warpSize;

        while ((ptsIdx >= endIdx) && (ptsIdx < warpEndIdx))
        {
            if (++cellIdx >= effectiveNodes) break;
            startIdx = endIdx;
            endIdx = s_ptIdxs[cellIdx + 1];
            center = s_Centers[cellIdx];
        }
    }
    // Wait for all threads in warp to finish
    warp.sync();

	// Write back to global memory
	for (int i = 0; i < effectiveNodes; ++i)
	{
        const int nodeIdx = nodes[firstIndexWarp + i];
		const int nodeOffset = nodeIdx * p * p;

		for (int j = warp.thread_rank(); j < p * p; j += 32)
		{
			M[nodeOffset + j] = s_M[i * coefsExpansion + j];
		}
	}
}

P2MInfo::P2MInfo(size_t capacity)
{
    nodes_.reserve(capacity);
	depths_.reserve(capacity);
    pointsEnd_.reserve(capacity);
    centers_.reserve(capacity);
}

P2MInfo::~P2MInfo()
{
    if (dev_nodes_ != nullptr)
    {
        cudaFree(dev_nodes_);
        cudaFree(dev_pointsEnd_);
        cudaFree(dev_centers_);
    }
}

void P2MInfo::add(vpm::nidx_t nodeIndex, int depth, vpm::pidx_t pointEndIndex, const vpm::vec3& nodeCenter)
{
    nodes_.push_back(nodeIndex);
	depths_.push_back(depth);
    pointsEnd_.push_back(pointEndIndex);
    centers_.push_back(nodeCenter);
}

void P2MInfo::addOffsets(const std::array<vpm::nidx_t, MAX_DEPTH>& depthOffsets)
{
    for (int i = 0; i < nodes_.size(); ++i)
    {
		nodes_[i] += depthOffsets[depths_[i]];
    }
}

void P2MInfo::toDevice() const
{
	if (dev_nodes_ != nullptr) return;

    const size_t count = nodes_.size();

    cudaMalloc((void**)&dev_nodes_,       sizeof(vpm::nidx_t) * count);
    cudaMalloc((void**)&dev_pointsEnd_,   sizeof(vpm::pidx_t) * count);
    cudaMalloc((void**)&dev_centers_,     sizeof(vpm::vec3) * count);

    cudaMemcpy(dev_nodes_,       nodes_.data(),       sizeof(vpm::nidx_t) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pointsEnd_,   pointsEnd_.data(),   sizeof(vpm::pidx_t) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_centers_,     centers_.data(),     sizeof(vpm::vec3) * count, cudaMemcpyHostToDevice);
}