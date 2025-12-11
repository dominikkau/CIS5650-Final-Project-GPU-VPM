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

// Set up P2M kernel testing environment
void testP2MKernel()
{
	// Define the number of repetitions for timing
    const unsigned int REPETITIONS = 10000;
	// Define block size
    const unsigned int blockSize = 128;
	// Define threads per cell
	const unsigned int threadsPerCell = 32;
	// Number of multipole expansion terms
    const int p = 10;
    // Number of FMM levels
    const int depth = 8;

	// Define vortex rings properties
    unsigned int numParticles = 0;
    std::vector<VortexRing> vortexRings(2);
    const float dz = 0.7906f;
	for (int i = 0; i < vortexRings.size(); i++)
	{
        vortexRings[i].circulation = 1.0f;
        vortexRings[i].R = 0.7906f;
        vortexRings[i].Rcross = 0.07906f;
        vortexRings[i].sigma = 0.07906f;
        vortexRings[i].Nphi = 200;
        vortexRings[i].nc = 3;
        vortexRings[i].position = { 0, 0, dz * i };
		numParticles += numberParticles(vortexRings[i]);
	}

	std::cout << "Total number of particles: " << numParticles << std::endl;

    // Particles per cell at the deepest level
    int numCells = 1 << depth;
	int particlesPerCell = (numParticles + numCells - 1) / numCells;
    
    // Compute size of required shared memory
	size_t sharedMemSize = blockSize/threadsPerCell * p * p * sizeof(float);
	// Calculate grid size
	int numBlocks = (numParticles + blockSize - 1) / blockSize;

	// Compute vortex ring particle positions and strengths
    ParticleBuffer particleBuffer{ ParticleBufferType::HOST };
    int inputBufferMask = BufferField::X | BufferField::INDEX | BufferField::GAMMA | BufferField::SIGMA;
    particleBuffer.mallocFields(numParticles, inputBufferMask);

	initVortexRings(particleBuffer, numParticles);

    // Initialize FMM input buffer on host
	float *fmm_buffer = new float[numParticles * 4];
    for (int i = 0; i < numParticles; i++)
    {
        fmm_buffer[i * 4 + 0] = 1.0f; // particleBuffer.X[i].x;
        fmm_buffer[i * 4 + 1] = 0.5f; // particleBuffer.X[i].y;
        fmm_buffer[i * 4 + 2] = 0.7f; // particleBuffer.X[i].z;
        fmm_buffer[i * 4 + 3] = 0.1f; // particleBuffer.Gamma[i].x;
    }
    particleBuffer.freeFields();

	// Initialize FMM input buffer on device
	float* d_fmm_buffer;
	cudaMalloc((void**)&d_fmm_buffer, numParticles * 4 * sizeof(float));

    // Initialize FMM output buffer on host
	float* fmm_M_buffer = new float[numCells * (p * p)];

	// Initialize FMM output buffer on device
	float* d_fmm_M_buffer;
	cudaMalloc((void**)&d_fmm_M_buffer, numCells * (p * p) * sizeof(float));

	// Copy data to device
	cudaMemcpy(d_fmm_buffer, fmm_buffer, numParticles * 4 * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Launching kernel" << std::endl;
    printf("numBlocks %d, blockSize %d, sharedMemSize %d, numCells %d, particlesPerCell %d\n", numBlocks, blockSize, sharedMemSize, numCells, particlesPerCell);

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

	// Launch the kernel multiple times
    #pragma unroll
    for (int i = 0; i < REPETITIONS; i++)
    {
		// Your kernel launch code here
        fmmP2M<threadsPerCell><<<numBlocks, blockSize, sharedMemSize>>>(reinterpret_cast<vpmvec4*>(d_fmm_buffer), numParticles, d_fmm_M_buffer, p, depth);
        checkCUDAError("fmm_P2M failed!");
        cudaDeviceSynchronize();
    }

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output the duration
    std::cout << "Kernel execution took " << milliseconds / REPETITIONS << " ms on average" << std::endl;

    cudaMemcpy(fmm_M_buffer, d_fmm_M_buffer, numCells * p * p * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16; i++)
    {
		printf("M[%d]: %f\n", i, fmm_M_buffer[i]);
    }

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	// Free host memory
	delete[] fmm_buffer;
    delete[] fmm_M_buffer;
	// Free device memory
	cudaFree(d_fmm_buffer);
    cudaFree(d_fmm_M_buffer);
}

__host__ __device__ int point_index_analytic(int index, int level, int point_count)
{
    return (int)floorf((float)point_count * index / (1 << level));
}

__device__ __forceinline__ int ilog2(unsigned int x) {
    return 31 - __clz(x);
}

// Computes Multipole expansion of particles
// Evaluates regular spherical basis function
template<unsigned int threads_per_cell>
__global__ void fmmP2M(vpmvec4* xqs, int N, float* Rout, int p, int depth)
{
    auto block = cg::this_thread_block();
    auto cell = cg::tiled_partition<threads_per_cell>(block);

    //const int idx = block.group_index().x * block.size() + block.thread_rank();
    //if (idx >= threads_per_cell * (1 << depth)) return;

	const int globalCellIdx = block.group_index().x * cell.meta_group_size() + cell.meta_group_rank();
    const int coefsPerBlock = p * p * cell.meta_group_size();

	// Zero initialize shared memory
    extern __shared__ float sh[];
    for (int i = threadIdx.x; i < coefsPerBlock; i += block.size())
    {
        sh[i] = 0.0f;
    }

    if (globalCellIdx >= (1 << depth)) return; // Out of range

	// Pointer to expansion coefficients of this cell in shared memory
    float* s_M = sh + p * p * cell.meta_group_rank();

	// Synchronize block to ensure shared memory is initialized
    block.sync();


    const int cellPtsIdxStart = point_index_analytic(globalCellIdx, depth, N);
    const int cellPtsIdxEnd = point_index_analytic(globalCellIdx + 1, depth, N);

    for (int cellPtsIdx = cellPtsIdxStart; cellPtsIdx < cellPtsIdxEnd; cellPtsIdx += threads_per_cell)
    {
        // Load position and charge data
		const int ptsIdx = cellPtsIdx + cell.thread_rank();
        vpmvec4 xq;
        if (ptsIdx < cellPtsIdxEnd)
            xq = xqs[ptsIdx];
        else
			xq = vpmvec4{ 0.0f, 0.0f, 0.0f, 0.0f };

        vpmfloat x = xq.x;
        vpmfloat y = xq.y;
        vpmfloat z = xq.z;
        vpmfloat q = xq.w;
        vpmfloat r2 = x * x + y * y + z * z;

        float Rnm1mm1pos =      1.0f; // R_{n-1}^{m-1}
        float Rnmneg     =  0.5f * y; // R_{n}^{-m}
        float Rnmm1pos   =        -z; // R_{n}^{m-1}
        float Rnmpos     = -0.5f * x; // R_{n}^{m}

		float Rnm1mm1neg;   // R_{n-1}^{-(m-1)}
		float Rnmm1neg;     // R_{n}^{-(m-1)}  
		float Rnp1mm1neg;   // R_{n+1}^{-(m-1)}
		float Rnp1mm1pos;   // R_{n+1}^{m-1}

        s_M[0] += cg::reduce(cell, q * Rnm1mm1pos, cg::plus<float>());
        s_M[1] += cg::reduce(cell, q * Rnmneg, cg::plus<float>());
        s_M[2] += cg::reduce(cell, q * Rnmm1pos, cg::plus<float>());
        s_M[3] += cg::reduce(cell, q * Rnmpos, cg::plus<float>());

        for (int n_ = 2; n_ < p; n_++)
        {
            float facz = z * (2.0f * n_ - 1.0f);
            Rnp1mm1pos = -(facz * Rnmm1pos + r2 * Rnm1mm1pos) / (n_ * n_);
            s_M[n_ * n_ + n_] += cg::reduce(cell, q * Rnp1mm1pos, cg::plus<float>());
            Rnm1mm1pos = Rnmm1pos;
            Rnmm1pos = Rnp1mm1pos;
        }

        Rnm1mm1neg = Rnmneg;
        Rnm1mm1pos = Rnmpos;

        for (int n = 2; n < p; n++)
        {
            float div = -0.5f / n;
            Rnmneg = div * (x * Rnm1mm1neg - y * Rnm1mm1pos);
			Rnmpos = div * (x * Rnm1mm1pos + y * Rnm1mm1neg);

			// Save to shared memory
            s_M[n * n] += cg::reduce(cell, q * Rnmneg, cg::plus<float>());
            s_M[n * n + 2 * n] += cg::reduce(cell, q * Rnmpos, cg::plus<float>());

            Rnmm1neg = -z * Rnm1mm1neg;
            Rnmm1pos = -z * Rnm1mm1pos;

            // Save to shared memory
            s_M[n * n + 1] += cg::reduce(cell, q * Rnmm1neg, cg::plus<float>());
            s_M[n * n + 2 * n - 1] += cg::reduce(cell, q * Rnmm1pos, cg::plus<float>());

            int m = n - 1;
            int m2 = m * m;
			for (int n_ = n + 1; n_ < p; n_++)
			{
                float facz = z * (2 * n_ - 1);
				div = -1.0f / (n_ * n_ - m2);
				Rnp1mm1neg = (facz * Rnmm1neg + r2 * Rnm1mm1neg) * div;
                Rnp1mm1pos = (facz * Rnmm1pos + r2 * Rnm1mm1pos) * div;

                // Save to shared memory
                s_M[n_ * n_ + 2] += cg::reduce(cell, q * Rnp1mm1neg, cg::plus<float>());
                s_M[n_ * n_ + 2 * n_ - 2] += cg::reduce(cell, q * Rnp1mm1pos, cg::plus<float>());

                Rnm1mm1neg = Rnmm1neg;
                Rnm1mm1pos = Rnmm1pos;
                Rnmm1neg = Rnp1mm1neg;
                Rnmm1pos = Rnp1mm1pos;
			}

            Rnm1mm1neg = Rnmneg;
            Rnm1mm1pos = Rnmpos;
        }
    }

    block.sync();

	// Write back to global memory
    // Only use active threads
    const int validCellsInBlock = min(cell.meta_group_size(),
        (1 << depth) - block.group_index().x * cell.meta_group_size());
    const int coefsToWrite = p * p * validCellsInBlock;
	const int activeThreads = threads_per_cell * validCellsInBlock;

    for (int i = threadIdx.x; i < coefsToWrite; i += activeThreads)
    {
        Rout[blockIdx.x * coefsPerBlock + i] = sh[i];
    }
}