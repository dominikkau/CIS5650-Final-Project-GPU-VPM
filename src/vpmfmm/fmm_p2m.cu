#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "../vortexringsimulation.hpp"
#include "fmm_p2m.h"
#include <device_launch_parameters.h>

#define MAXP 32
#define THREADS_PER_CELL 32
#define THREADS_PER_CELL_B 5

// Set up P2M kernel testing environment
void testP2MKernel()
{
	// Define the number of repetitions for timing
    constexpr int REPETITIONS = 100;
    int blockSize = 32;
	// Number of multipole expansion terms
    int p = 10;
    // Number of FMM levels
    int depth = 8;

	if (p > MAXP)
	{
		std::cerr << "Error: p exceeds MAXP" << std::endl;
		return;
	}

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
    int threadsPerCell = 2;
    
    // Compute size of required shared memory
	size_t sharedMemSize = blockSize/THREADS_PER_CELL * p * p * sizeof(float);
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
		fmm_P2M<<<numBlocks, blockSize, sharedMemSize>>>(reinterpret_cast<vpmvec4*>(d_fmm_buffer), numParticles, d_fmm_M_buffer, p, depth);
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

    for (int i = 0; i < 10; i++)
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

__device__ void inline addToM(float* buffer, float q, float* M, int n)
{
    float value;	
    for (int m = MAXP - 1 - n; m <= MAXP - 1 + n; m++)
	{
        value = q * buffer[m];
        #pragma unroll
        for (int i = 1; i < THREADS_PER_CELL; i *= 2)
        {
            value += __shfl_xor_sync(0xFFFFFFFF, value, i);
        }
        M[n * n + m - (MAXP - 1 - n)] += value;
	}
}

// Computes Multipole expansion of particles
// Evaluates regular spherical basis function
__global__ void fmm_P2M(vpmvec4* xqs, int N, float* Rout, int p, int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int globalCellIdx = idx >> THREADS_PER_CELL_B;
	const int cellThreadIdx = idx & (THREADS_PER_CELL - 1);
    const int blockCellIdx = threadIdx.x >> THREADS_PER_CELL_B;

    const int ptsIdxStart = point_index_analytic(  globalCellIdx, depth, N);
    const int ptsIdxEnd   = point_index_analytic(1+globalCellIdx, depth, N);

    extern __shared__ float sh[];
    float* s_M = sh + p * p * blockCellIdx;

    float Rm1Buffer[2 * MAXP - 1];
    float Rm2Buffer[2 * MAXP - 1];
    float *Rm1BufferPtr = Rm1Buffer;
	float *Rm2BufferPtr = Rm2Buffer;

    for (int i = idx; i < p * p * blockDim.x / THREADS_PER_CELL; i += blockDim.x)
    {
        sh[i] = 0.0f;
	}

    __syncthreads();

    for (int ptsIdx = ptsIdxStart; ptsIdx < ptsIdxEnd + THREADS_PER_CELL - 1; ptsIdx += THREADS_PER_CELL)
	{
		// Load xq
		vpmvec4 xq;
        if ((ptsIdx + cellThreadIdx >= ptsIdxEnd) || (ptsIdx >= N))
			xq = vpmvec4(0.0f); // Dummy point does not contribute to sum
        else
        {
            xq = xqs[ptsIdx + cellThreadIdx];
        }
        vpmfloat x = xq.x;
        vpmfloat y = xq.y;
        vpmfloat z = xq.z;
        vpmfloat q = xq.w;

        vpmfloat r2 = x * x + y * y + z * z;

        Rm2BufferPtr[MAXP - 1] = 1.0f;      // R_0^0
		addToM(Rm2BufferPtr, q, s_M, 0);

        Rm1BufferPtr[MAXP - 2] = 0.5f * y;  // R_1^{-1}
        Rm1BufferPtr[MAXP - 1] = -z;        // R_1^0  
        Rm1BufferPtr[MAXP    ] = -0.5f * x; // R_1^1
        addToM(Rm1BufferPtr, q, s_M, 1);

        for (int n = 2; n < p; n++)
        {
            float div = -0.5f / n;
            Rm2BufferPtr[MAXP - 1 + n] = div * (x * Rm1BufferPtr[MAXP + n - 2] + y * Rm1BufferPtr[MAXP - n    ]);
            Rm2BufferPtr[MAXP - 1 - n] = div * (x * Rm1BufferPtr[MAXP - n    ] - y * Rm1BufferPtr[MAXP + n - 2]);

            Rm2BufferPtr[MAXP + n - 2] = -z * Rm1BufferPtr[MAXP + n - 2];
            Rm2BufferPtr[MAXP - n    ] = -z * Rm1BufferPtr[MAXP - n    ];

            float facz = z * (2.0f * n - 1.0f);
            Rm2BufferPtr[MAXP - 1] = -(facz * Rm1BufferPtr[MAXP - 1] + r2 * Rm2BufferPtr[MAXP - 1]) / (n * n);
            for (int m = 1; m < n - 1; m++)
            {
                div = -1.0f / ((n - m) * (n + m));
                Rm2BufferPtr[MAXP - 1 + m] = (facz * Rm1BufferPtr[MAXP - 1 + m] + r2 * Rm2BufferPtr[MAXP - 1 + m]) * div;
                Rm2BufferPtr[MAXP - 1 - m] = (facz * Rm1BufferPtr[MAXP - 1 - m] + r2 * Rm2BufferPtr[MAXP - 1 - m]) * div;
            }

			addToM(Rm2Buffer, q, s_M, n);
            d_swap(Rm1BufferPtr, Rm2BufferPtr);
        }
	}

    __syncthreads();

	const int cellsPerBlock = blockDim.x / THREADS_PER_CELL;
	const int coefsPerBlock = p * p * cellsPerBlock;

    for (int i = idx; i < p * p * blockDim.x / THREADS_PER_CELL; i += blockDim.x)
    {
        Rout[blockIdx.x * coefsPerBlock + i] = sh[i];
    }
}