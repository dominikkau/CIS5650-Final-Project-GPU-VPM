#include <cuda.h>
#include <iostream>
#include <vector>
#include "../vortexringsimulation.hpp"
#include "fmm_p2m.h"
#include <device_launch_parameters.h>

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

// Computes Multipole expansion of particles
// Evaluates regular spherical basis function
__global__ void fmm::p2m(const vpm::nidx_t* nodes, const vpm::pidx_t* pointsEnd, const vpm::vec3* centers, size_t count, const vpm::vec3* xs, const vpm::real* qs, float* M, int p)
{
    const size_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

 //   auto block = cg::this_thread_block();

 //   extern __shared__ float sh[];

 //   size_t* s_pointsEnd = (size_t*)sh;

	//if (globalIdx < count)  s_pointsEnd[1 + threadIdx.x] = pointsEnd[globalIdx];
	//
 //   if (threadIdx.x == 0)
 //   {
 //       if (globalIdx == 0) s_pointsEnd[0] = 0;
 //       else                s_pointsEnd[0] = pointsEnd[globalIdx - 1];
 //   }

 //   auto nodeGroup = cg::labeled_partition<size_t>(block, 10ULL);

    //cg::labeled_partition
    //auto cell = cg::tiled_partition<threadGroupSize>(block);

    //int cellsPerGro = 8;

    //const int threadGroupIndex = block.group_index().x * cell.meta_group_size() + cell.meta_group_rank();

	//const int globalCellIdx = block.group_index().x * cell.meta_group_size() + cell.meta_group_rank();
    //const int coefsPerBlock = p * p * cell.meta_group_size();

	


	const int coefsExpansion = p * p;
    const int coefsPerBlock = coefsExpansion * blockDim.x;

    const int laneId = threadIdx.x & (warpSize - 1);
	const int warpId = threadIdx.x >> 5;
	const int coefsPerWarp = coefsExpansion * warpSize;

	// Zero initialize shared memory
    extern __shared__ float sh[];
    const int upperBound = warpId * coefsPerWarp + coefsPerWarp;
    for (int i = warpId * coefsPerWarp + laneId; i < upperBound; i += warpSize)
    {
        sh[i] = 0.0f;
    }
    __syncwarp();

    if (globalIdx >= count) return; // Out of range


	// Pointer to expansion coefficients of this cell in shared memory
    //volatile float* const s_M = sh + p * p * cell.meta_group_rank();
    float* const s_M = sh + coefsExpansion * threadIdx.x;

	// Synchronize block to ensure shared memory is initialized
    //block.sync();

    const vpm::nidx_t cellIdx  = nodes[globalIdx];
    const vpm::pidx_t startIdx = globalIdx == 0 ? 0 : pointsEnd[globalIdx - 1];
    const vpm::pidx_t endIdx   = pointsEnd[globalIdx];

    //const int cellPtsIdxStart = point_index_analytic(globalCellIdx, depth, N);
    //const int cellPtsIdxEnd   = point_index_analytic(globalCellIdx + 1, depth, N);

	const vpm::real xc = centers[globalIdx].x;
	const vpm::real yc = centers[globalIdx].y;
	const vpm::real zc = centers[globalIdx].z;

    for (vpm::pidx_t ptsIdx = startIdx; ptsIdx < endIdx; ++ptsIdx) // TODO: This is exactly how not to do it...
    {
        // Load position and charge data
		//const int ptsIdx = cellPtsIdx + cell.thread_rank();
   
        //vpm::real x;
		//vpm::real y;
		//vpm::real z;
		//vpm::real q;
        const vpm::real x = xs[ptsIdx].x - xc;
        const vpm::real y = xs[ptsIdx].y - yc;
        const vpm::real z = xs[ptsIdx].z - zc;
        const vpm::real q = qs[ptsIdx];

        const vpm::real r2 = x * x + y * y + z * z;

		// Define initial values for recurrence
        float Rnm1mm1pos =      1.0f; // R_{n-1}^{m-1}
        float Rnmneg     = -0.5f * y; // R_{n}^{-m}
        float Rnmm1pos   =         z; // R_{n}^{m-1}
        float Rnmpos     =  0.5f * x; // R_{n}^{m}
        
		// Save to shared memory
        //s_M[0] += cg::reduce(cell, q * Rnm1mm1pos, cg::plus<float>());
        //s_M[1] += cg::reduce(cell, q * Rnmneg,     cg::plus<float>());
        //s_M[2] += cg::reduce(cell, q * Rnmm1pos,   cg::plus<float>());
        //s_M[3] += cg::reduce(cell, q * Rnmpos,     cg::plus<float>());
        s_M[0] += q * Rnm1mm1pos;
        s_M[1] += q * Rnmneg;
        s_M[2] += q * Rnmm1pos;
        s_M[3] += q * Rnmpos;

        float Rnm1mm1neg;   // R_{n-1}^{-(m-1)}
        float Rnmm1neg;     // R_{n}^{-(m-1)}  
        //float Rnp1mm1neg;   // R_{n+1}^{-(m-1)}
        //float Rnp1mm1pos;   // R_{n+1}^{m-1}


        for (int n_ = 2; n_ < p; n_++)
        {
            const int n_2 = n_ * n_;
            const float facz = z * (2.0f * n_ - 1.0f);
            const float Rnp1mm1pos = (facz * Rnmm1pos - r2 * Rnm1mm1pos) / n_2;
            //s_M[n_2 + n_] += cg::reduce(cell, q * Rnp1mm1pos, cg::plus<float>());
            s_M[n_2 + n_] += q * Rnp1mm1pos;
            Rnm1mm1pos = Rnmm1pos;
            Rnmm1pos = Rnp1mm1pos;
        }

        Rnm1mm1neg = Rnmneg;
        Rnm1mm1pos = Rnmpos;

        for (int n = 2; n < p; n++)
        {
			const int n2 = n * n;
            const float div = 0.5f / n;
            Rnmneg = div * (x * Rnm1mm1neg - y * Rnm1mm1pos);
			Rnmpos = div * (x * Rnm1mm1pos + y * Rnm1mm1neg);

			// Save to shared memory
            //s_M[n2        ] += cg::reduce(cell, q * Rnmneg, cg::plus<float>());
            //s_M[n2 + 2 * n] += cg::reduce(cell, q * Rnmpos, cg::plus<float>());
            s_M[n2] += q * Rnmneg;
            s_M[n2 + 2 * n] += q * Rnmpos;

            Rnmm1neg = z * Rnm1mm1neg;
            Rnmm1pos = z * Rnm1mm1pos;

            // Save to shared memory
            //s_M[n2 + 1        ] += cg::reduce(cell, q * Rnmm1neg, cg::plus<float>());
            //s_M[n2 + 2 * n - 1] += cg::reduce(cell, q * Rnmm1pos, cg::plus<float>());
            s_M[n2 + 1] += q * Rnmm1neg;
            s_M[n2 + 2 * n - 1] += q * Rnmm1pos;

            const int m = n - 1;
            const int m2 = m * m;
			for (int n_ = n + 1; n_ < p; n_++)
			{
				const int n_2 = n_ * n_;
                const float facz = z * (2 * n_ - 1);
				const float div_ = 1.0f / (n_2 - m2);
				const float Rnp1mm1neg = (facz * Rnmm1neg - r2 * Rnm1mm1neg) * div_;
                const float Rnp1mm1pos = (facz * Rnmm1pos - r2 * Rnm1mm1pos) * div_;

                // Save to shared memory
                //s_M[n_2 + 2         ] += cg::reduce(cell, q * Rnp1mm1neg, cg::plus<float>());
                //s_M[n_2 + 2 * n_ - 2] += cg::reduce(cell, q * Rnp1mm1pos, cg::plus<float>());
                s_M[n_2 + 2] += q * Rnp1mm1neg;
                s_M[n_2 + 2 * n_ - 2] += q * Rnp1mm1pos;

                Rnm1mm1neg = Rnmm1neg;
                Rnm1mm1pos = Rnmm1pos;
                Rnmm1neg = Rnp1mm1neg;
                Rnmm1pos = Rnp1mm1pos;
			}

            Rnm1mm1neg = Rnmneg;
            Rnm1mm1pos = Rnmpos;
        }
    }

    __syncwarp();

	// Write back to global memory
    const int cellsPerWarp = warpSize;
	for (int i = 0; i < cellsPerWarp; ++i)
	{
        const int cellIdx = nodes[blockIdx.x * blockDim.x + cellsPerWarp * warpId + i];
		const int cellOffset = cellIdx * p * p;
		const int cellOffsetLocal = warpId * coefsPerWarp + i * p * p;

		for (int j = laneId; j < p * p; j += warpSize)
		{
			M[cellOffset + j] = sh[cellOffsetLocal + j];
		}
	}

    //for (int i = warpId * coefsPerWarp + laneId; i < upperBound; i += warpSize)
    //{
    //    M[blockOffset + i] = sh[i];
    //}
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