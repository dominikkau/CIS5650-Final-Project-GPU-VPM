#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <random>
#include <memory>
#include "vpmmain.h"
#include "../lean_vtk.hpp"
#include "../vortexringsimulation.hpp"
#include "../roundjetsimulation.hpp"
#include <device_launch_parameters.h>

void calcEstrNaiveWrapper(CUDAKernelParams params, int targetN, int sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, KernelType kernel, bool reset, vpm::real testFilterFactor)
{
    switch (kernel)
    {
    case KernelType::SINGULAR:
        calcEstrNaive<<<params.numBlocks, params.blockSize, params.sharedBytes, params.stream>>>(targetN, sourceN, targetParticles, sourceParticles, SingularKernel(), reset, testFilterFactor);
        break;
    case KernelType::GAUSSIAN:
        calcEstrNaive<<<params.numBlocks, params.blockSize, params.sharedBytes, params.stream>>>(targetN, sourceN, targetParticles, sourceParticles, GaussianKernel(), reset, testFilterFactor);
        break;
    case KernelType::GAUSSIAN_ERF:
        calcEstrNaive<<<params.numBlocks, params.blockSize, params.sharedBytes, params.stream>>>(targetN, sourceN, targetParticles, sourceParticles, GaussianErfKernel(), reset, testFilterFactor);
        break;
    case KernelType::WINCKELMAN:
        calcEstrNaive<<<params.numBlocks, params.blockSize, params.sharedBytes, params.stream>>>(targetN, sourceN, targetParticles, sourceParticles, WinckelmansKernel(), reset, testFilterFactor);
        break;
    default:
        // Default to GaussianKernel if unknown type
        calcEstrNaive<<<params.numBlocks, params.blockSize, params.sharedBytes, params.stream>>>(targetN, sourceN, targetParticles, sourceParticles, GaussianKernel(), reset, testFilterFactor);
        break;
    }
}

template <typename K>
__global__ void calcEstrNaive(int targetN, int sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, K kernel, bool reset, vpm::real testFilterFactor) {

    const int index = threadIdx.x + (blockIdx.x * blockDim.x);

    const int s_index = threadIdx.x;
    // number of vpm::reals per particle: 3 + 9 + 3 + 1 = 16
    extern __shared__ vpm::real sharedMemory[];
    vpm::vec3* s_sourceX = (vpm::vec3*)sharedMemory;
    vpm::mat3* s_sourceJ = (vpm::mat3*)(s_sourceX + blockDim.x);
    vpm::vec3* s_sourceGammaSigma = (vpm::vec3*)(s_sourceJ + blockDim.x);
    vpm::real* s_sourceInvSigma = (vpm::real*)(s_sourceGammaSigma + blockDim.x);

    // Get required variables from global memory
    vpm::vec3 targetX;
    vpm::mat3 targetJ;
    vpm::vec3 targetSFS;
    if (index < targetN) {
        targetX = targetParticles.X()[index];
        targetJ = targetParticles.J()[index];
        if (reset) {
            targetSFS = vpm::vec3{ 0.0f };
        }
        else {
            targetSFS = targetParticles.SFS()[index];
        }
    }
    else {
        targetX = vpm::vec3{ 0.0f };
        targetJ = vpm::mat3{ 0.0f };
        targetSFS = vpm::vec3{ 0.0f };
    }

    vpm::vec3 targetXSigma{ 0.0f };
    for (int j = 0; j < sourceN; j += blockDim.x) {
        if (j + s_index < sourceN) {
            s_sourceInvSigma[s_index] = 1.0f / (sourceParticles.sigma()[s_index + j] * testFilterFactor);
            s_sourceX[s_index] = sourceParticles.X()[s_index + j] * s_sourceInvSigma[s_index];
            s_sourceJ[s_index] = sourceParticles.J()[s_index + j];
            s_sourceGammaSigma[s_index] = sourceParticles.Gamma()[s_index + j]
                * s_sourceInvSigma[s_index] * s_sourceInvSigma[s_index] * s_sourceInvSigma[s_index];
            targetXSigma = targetX * s_sourceInvSigma[s_index];
        }
        __syncthreads();

        for (int i = 0; (i < blockDim.x) && (j + i < sourceN); ++i) {
            targetSFS += kernel.zeta(glm::length(targetXSigma - s_sourceX[i]))
                * xDotNablaY(s_sourceGammaSigma[i], targetJ - s_sourceJ[i]);
        }

        __syncthreads();
    }

    // Copy variables back to global memory
    if (index < targetN) {
        targetParticles.SFS()[index] = targetSFS;
    }
}

void calcVelJacNaiveWrapper(CUDAKernelParams params, int targetN, int sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, KernelType kernel, bool reset, vpm::real testFilterFactor)
{
    switch (kernel)
    {
    case KernelType::SINGULAR:
        calcVelJacNaive<<<params.numBlocks, params.blockSize, params.sharedBytes, params.stream>>>(targetN, sourceN, targetParticles, sourceParticles, SingularKernel(), reset, testFilterFactor);
        break;
    case KernelType::GAUSSIAN:
        calcVelJacNaive<<<params.numBlocks, params.blockSize, params.sharedBytes, params.stream>>>(targetN, sourceN, targetParticles, sourceParticles, GaussianKernel(), reset, testFilterFactor);
        break;
    case KernelType::GAUSSIAN_ERF:
        calcVelJacNaive<<<params.numBlocks, params.blockSize, params.sharedBytes, params.stream>>>(targetN, sourceN, targetParticles, sourceParticles, GaussianErfKernel(), reset, testFilterFactor);
        break;
    case KernelType::WINCKELMAN:
        calcVelJacNaive<<<params.numBlocks, params.blockSize, params.sharedBytes, params.stream>>>(targetN, sourceN, targetParticles, sourceParticles, WinckelmansKernel(), reset, testFilterFactor);
        break;
    default:
        // Default to GaussianKernel if unknown type
        calcVelJacNaive<<<params.numBlocks, params.blockSize, params.sharedBytes, params.stream>>>(targetN, sourceN, targetParticles, sourceParticles, GaussianKernel(), reset, testFilterFactor);
        break;
    }
}

template <typename K>
__global__ void calcVelJacNaive(int targetN, int sourceN, ParticleBuffer targetParticles, 
    ParticleBuffer sourceParticles, K kernel, bool reset, vpm::real testFilterFactor) {

    const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	const vpm::real invTestFilterFactor = 1.0f / testFilterFactor;

    const int s_index = threadIdx.x;
    extern __shared__ vpm::real sharedMemory[];
    vpm::vec3*  s_sourceX     = (vpm::vec3*)sharedMemory;
    vpm::vec3*  s_sourceGamma = (vpm::vec3*)(s_sourceX + blockDim.x);
    vpm::real* s_sourceInvSigma = (vpm::real*)(s_sourceGamma + blockDim.x);

    vpm::vec3 targetX;
    vpm::vec3 targetU;
    vpm::mat3 targetJ;
    if (index < targetN) {
        // Get target variables from global memory
        targetX = targetParticles.X()[index];

        if (reset) {
            targetU = vpm::vec3{ 0.0f };
            targetJ = vpm::mat3{ 0.0f };
        }
        else {
            targetU = targetParticles.U()[index];
            targetJ = targetParticles.J()[index];
        }
    }
    else {
		targetX = vpm::vec3{ 0.0f };
		targetU = vpm::vec3{ 0.0f };
		targetJ = vpm::mat3{ 0.0f };
    }

    // Copy source variables into shared memory
    for (int j = 0; j < sourceN; j += blockDim.x) {
        if (j + s_index < sourceN) {
            s_sourceX[s_index]     = sourceParticles.X()[s_index + j];
            s_sourceGamma[s_index] = sourceParticles.Gamma()[s_index + j];
            s_sourceInvSigma[s_index] = invTestFilterFactor / sourceParticles.sigma()[s_index + j];
        }
        __syncthreads();

        for (int i = 0; (i < blockDim.x) && (j + i < sourceN); ++i) {
            vpm::vec3 dX = targetX - s_sourceX[i];
            vpm::real r = glm::length(dX);
            const vpm::real invSourceSigma = r * s_sourceInvSigma[i];
			vpm::vec3 sourceGamma = s_sourceGamma[i];
            
            if (r == 0.0f) continue;
            const vpm::real invR = 1.0f / r;

            // Kernel evaluation
			const vpm::vec2 g_dgdr = kernel.g_dgdr(invSourceSigma);

            const vpm::real tmp = -const4 * (invR * invR * invR);

            // Compute velocity
            const vpm::vec3 crossProd = tmp * glm::cross(dX, sourceGamma);
            targetU += g_dgdr[0] * crossProd;

            // Compute Jacobian
            dX *= (g_dgdr[1] * invSourceSigma - 3.0f * g_dgdr[0]) * (invR * invR);

            targetJ += glm::outerProduct(crossProd, dX);
            sourceGamma *= tmp * g_dgdr[0];

            // Account for kronecker delta term
            targetJ[0][1] -= sourceGamma[2];
            targetJ[0][2] += sourceGamma[1];
            targetJ[1][0] += sourceGamma[2];
            targetJ[1][2] -= sourceGamma[0];
            targetJ[2][0] -= sourceGamma[1];
            targetJ[2][1] += sourceGamma[0];
        }

        __syncthreads();
    }

    if (index < targetN) {
        // Copy variables back to global memory
        targetParticles.U()[index] = targetU;
        targetParticles.J()[index] = targetJ;
    }
}

__global__ void rungeKuttaStep(int N, ParticleBuffer particles, vpm::real a, vpm::real b, vpm::real dt, vpm::real zeta0, vpm::vec3 Uinf) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    const vpm::real particleC   = particles.C()[index][0];
    const vpm::vec3  particleU   = particles.U()[index];
    const vpm::vec3  particleSFS = particles.SFS()[index];
    const vpm::mat3  particleJ   = particles.J()[index];
    
    vpm::real particleSigma = particles.sigma()[index];
    vpm::vec3  particleGamma = particles.Gamma()[index];
    vpm::vec3  particleX     = particles.X()[index];
    vpm::mat3  particleM;
    if (a == 1.0f || a == 0.0f) {
        particleM = vpm::mat3{ 0.0f };
    }
    else {
        particleM = particles.M()[index];
    }

    // Position update
    particleM[0] = a * particleM[0] + dt * (particleU + Uinf);
    particleX += b * particleM[0];
    particles.X()[index] = particleX;

    vpm::vec3 S = xDotNablaY(particleGamma, particleJ);
#ifdef CLASSIC_VPM
    vpm::real Z = 0.0f;
#else
    vpm::real Z = (vpm::real)0.2 * glm::dot(S, particleGamma) / glm::dot(particleGamma, particleGamma);
#endif

    // Gamma update
    particleM[1] = a * particleM[1] + dt * (S - 3.0f * Z * particleGamma
        - particleC * particleSFS * particleSigma * particleSigma * particleSigma / zeta0);
    particleGamma += b * particleM[1];
    particles.Gamma()[index] = particleGamma;

#ifndef CLASSIC_VPM
    // Sigma update
    particleM[2][1] = a * particleM[2][1] - dt * (particleSigma * Z);
    particleSigma += b * particleM[2][1];

    particles.sigma()[index] = particleSigma;
#endif

    particles.M()[index] = particleM;
}

void rungeKutta(ParticleField& field, vpm::real dt, bool useRelax, int numBlocks, int blockSize, cudaStream_t stream) {

    const vpm::real rungeKuttaCoefs[3][2] = {
        {0.0, 1.0 / 3.0},
        {-5.0 / 9.0, 15.0 / 16.0},
        {-153.0 / 128.0, 8.0 / 15.0}
    };

    const int N = field.numParticles;
    const Kernel* kernelPointer = getKernel(field.kernel);

    // Loop over the pairs
    for (int i = 0; i < 3; ++i) {
        vpm::real a = rungeKuttaCoefs[i][0];
        vpm::real b = rungeKuttaCoefs[i][1];

        // RUN SFS
        (*field.sfs)(field, a, b, numBlocks, blockSize, stream);

        rungeKuttaStep<<<numBlocks, blockSize, 0, stream>>>(N, field.dev_particles, a, b, dt, kernelPointer->zeta(0.0f), field.uInf);
        checkCUDAError("rungeKuttaStep failed!");
    }

    (*field.relaxation)(field, numBlocks, blockSize, stream);

    ++field.timeStep;
    field.synchronized = false;
    delete kernelPointer;
}

int outputMaskToBufferMask(int outputMask) {
    int bufferMask = 0;
    if (outputMask & OutputType::X) bufferMask |= BufferField::X;
    if (outputMask & OutputType::U) bufferMask |= BufferField::U;
    if (outputMask & OutputType::OMEGA) bufferMask |= BufferField::J;
    if (outputMask & OutputType::SIGMA) bufferMask |= BufferField::SIGMA;
    if (outputMask & OutputType::GAMMA) bufferMask |= BufferField::GAMMA;
    if (outputMask & OutputType::INDEX) bufferMask |= BufferField::INDEX;

    return bufferMask;
}

void writeVTK(ParticleBuffer &particles, size_t N, const std::string& filename, int outputMask) {
    const int dim = 3;

    static leanvtk::VTUWriter writer;

    static std::vector<double> particleX;
    static std::vector<double> particleU;
    static std::vector<double> particleGamma;
    static std::vector<double> particleOmega;
    static std::vector<double> particleSigma;
    static std::vector<double> particleIdx;

    if (outputMask & OutputType::X) {
        particleX.insert(
            particleX.end(),
            (vpm::real*)particles.X(),
            (vpm::real*)(particles.X() + N)
        );

        writer.add_vector_field("position", particleX, dim);
    }
    if (outputMask & OutputType::U) {
        particleU.insert(
            particleU.end(),
            (vpm::real*)particles.U(),
            (vpm::real*)(particles.U() + N)
        );

        writer.add_vector_field("velocity", particleU, dim);
    }
    if (outputMask & OutputType::GAMMA) {
        particleGamma.insert(
            particleGamma.end(),
            (vpm::real*)particles.Gamma(),
            (vpm::real*)(particles.Gamma() + N)
        );

        writer.add_vector_field("circulation", particleGamma, dim);
    }
    if (outputMask & OutputType::SIGMA) {
        particleSigma.insert(
            particleSigma.end(),
            particles.sigma(),
            particles.sigma() + N
        );

        writer.add_scalar_field("sigma", particleSigma);
    }
    if (outputMask & OutputType::INDEX) {
        particleIdx.insert(
            particleIdx.end(),
            particles.index(),
            particles.index() + N
        );

        writer.add_scalar_field("index", particleIdx);
    }
    if (outputMask & OutputType::OMEGA) {
        particleOmega.reserve(N * dim);

        vpm::vec3 omega;
        for (int i = 0; i < N; ++i) {
            omega = nablaCrossX(particles.J()[i]);
            particleOmega.insert(particleOmega.end(), (vpm::real*)&omega, (vpm::real*)&omega + 3);
        }

        writer.add_vector_field("vorticity", particleOmega, dim);
    }

    writer.write_point_cloud("../output/" + filename + ".vtu", dim, particleX);
    writer.clear();

    particleX.clear();
    particleU.clear();
    particleGamma.clear();
    particleSigma.clear();
    particleIdx.clear();
    particleOmega.clear();
}

void calcVortexRingMetrics(ParticleField& field, int iteration, std::string filename, int numRings = 2) {
    field.syncParticlesDeviceToHost(BufferField::X | BufferField::GAMMA);
    int numParticlesRing = field.numParticles / numRings;

    std::vector<vpm::real> ringRadii;
    std::vector<vpm::vec3>  ringCenters;

    for (int j = 0; j < numRings; ++j) {
        int offset = j * numParticlesRing;
        // Calculate ring center
        vpm::vec3 ringCenter = vpm::vec3{ 0 };
        vpm::real totalGamma = 0;
        for (int i = offset; i < numParticlesRing + offset; ++i) {
            vpm::real Gamma = glm::length(field.particles.Gamma()[i]);
            totalGamma += Gamma;
            ringCenter += Gamma * field.particles.X()[i];
        }
        ringCenter /= totalGamma;

        // Calculate ring radius
        vpm::real ringRadius = 0;
        for (int i = offset; i < numParticlesRing + offset; ++i) {
            vpm::real Gamma = glm::length(field.particles.Gamma()[i]);
            vpm::real radius = glm::length(field.particles.X()[i] - ringCenter);
            ringRadius += Gamma * radius;
        }
        ringRadius /= totalGamma;

        ringCenters.push_back(ringCenter);
        ringRadii.push_back(ringRadius);
    }

    // Save results to csv file
    filename = "../output/" + filename + ".csv";
    std::ofstream file;
    if (iteration == 0) {
        // Overwrite file and write header in the first iteration
        file.open(filename, std::ios::out);
        if (file.is_open()) {
            file << "iteration";
            for (int i = 1; i < numRings + 1; ++i) {
                file << ",ring_center_" << i
                     << ",ring_radius_ " << i;
            }
            file << '\n';
        }
    }
    else {
        // Append to file in subsequent iterations
        file.open(filename, std::ios::app);
    }

    if (file.is_open()) {
        // Write iteration, ring center (Z-coordinate), and radius
        file << iteration;
        for (int i = 1; i < numRings + 1; ++i) {
            file << ',' << ringCenters[i-1][2]
                 << ',' << ringRadii[i-1];
        }
        file << '\n';
        file.close();
    }
    else {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
}

void runVPM(
    unsigned int maxParticles,
    unsigned int numParticles,
    unsigned int numTimeSteps,
    vpm::real dt,
    unsigned int fileSaveSteps,
    vpm::vec3 uInf,
    ParticleBuffer particleBuffer,
    RelaxationScheme *relaxation,
    SFSScheme *sfs,
    KernelType kernel,
    int blockSize,
    std::string filename) {

    int numBlocks = (numParticles + blockSize - 1) / blockSize;

    ParticleField field{
        particleBuffer,
        numParticles,
        0,
        kernel,
        uInf,
        std::unique_ptr<SFSScheme>(sfs),
        std::unique_ptr<RelaxationScheme>(relaxation)
    };

    int outputMask = OutputType::ALL;
    int bufferMask = outputMaskToBufferMask(outputMask);

    ParticleBuffer outputBufferHost{ ParticleBufferType::HOST_PINNED, maxParticles };
    outputBufferHost.mallocFields(bufferMask);

    for (int i = 0; i < numTimeSteps + 1; ++i) {
        // calcVortexRingMetrics(field, i, "test");

        rungeKutta(field, dt, true, numBlocks, blockSize);

        if ((fileSaveSteps != 0) && (i % fileSaveSteps == 0)) {
            writeVTK(outputBufferHost, field.numParticles, filename + "_" + std::to_string(field.timeStep), outputMask);

            std::cout << outputBufferHost.U()[0].x << std::endl;

            cpyParticleBuffer(outputBufferHost, field.dev_particles, bufferMask);

            cudaDeviceSynchronize();
        }
    }
}

void runBoundaryVPM(
    unsigned int maxParticles,
    unsigned int numParticles,
    unsigned int numBoundary,
    unsigned int numTimeSteps,
    vpm::real dt,
    unsigned int fileSaveSteps,
    vpm::vec3 uInf,
    ParticleBuffer particleBuffer,
    const ParticleBuffer boundaryBuffer,
    RelaxationScheme *relaxation,
    SFSScheme *sfs,
    KernelType kernel,
    int blockSize,
    std::string filename) {

    int numBlocks = (numParticles + blockSize - 1) / blockSize;

    int outputMask = OutputType::ALL;
    int bufferMask = outputMaskToBufferMask(outputMask);
    int boundaryMask = BufferField::X | BufferField::SIGMA | BufferField::GAMMA | BufferField::INDEX;

    ParticleBuffer dev_boundaryBuffer{ ParticleBufferType::DEVICE, numBoundary };
    dev_boundaryBuffer.mallocFields(boundaryMask);

    // Copy boundary particle buffer from host to device
    cpyParticleBuffer(dev_boundaryBuffer, boundaryBuffer, boundaryMask);

    ParticleField field{
        particleBuffer,
        numParticles,
        0,
        kernel,
        uInf,
        std::unique_ptr<SFSScheme>(sfs),
        std::unique_ptr<RelaxationScheme>(relaxation)
    };

    unsigned int boundaryIndex = numParticles;

    for (int i = 0; i < numTimeSteps; ++i) {
        std::cout << field.numParticles << " " << boundaryIndex << " " << field.particles.U()[0].x << std::endl;

        rungeKutta(field, dt, true, numBlocks, blockSize);

        if ((fileSaveSteps != 0) && (i % fileSaveSteps == 0)) {
            // writeVTK(field, filename, outputMask);

            std::cout << field.particles.U()[0].x << std::endl;

            field.syncParticlesDeviceToHost(bufferMask);
        }

        if (boundaryIndex + numBoundary >= field.particles.size()) boundaryIndex = 0;

        field.cpyParticlesDeviceToDevice(dev_boundaryBuffer, numBoundary, boundaryIndex, boundaryMask);

        numBlocks = (field.numParticles + blockSize - 1) / blockSize;

        boundaryIndex += numBoundary;
    }
}

void runSimulation() {
    // Define basic parameters
    unsigned int maxParticles = 50000;
    unsigned int numTimeSteps = 50;
    vpm::real dt = 1e-2;
    unsigned int numStepsVTK = 1;
    vpm::vec3 uInf{ 0, 0, 0 };
    int blockSize = 64;
    const int simulationType = 0;

    // Create host particle buffer
#ifdef PINNED_MEMORY
    ParticleBuffer particleBuffer{ ParticleBufferType::HOST_PINNED, maxParticles };
#else
    ParticleBuffer particleBuffer{ ParticleBufferType::HOST };
#endif
    int inputBufferMask = BufferField::X | BufferField::U | BufferField::J | BufferField::INDEX | BufferField::GAMMA | BufferField::SIGMA;
    particleBuffer.mallocFields(inputBufferMask);

    int numParticles;
    switch (simulationType)
    {
    case 0:
        numParticles = vortex_rings::initVortexRings(particleBuffer);
        break;

    case 1: {
        // Create host boundary buffer
        ParticleBuffer boundaryBuffer{ ParticleBufferType::HOST_PINNED, maxParticles };
        boundaryBuffer.mallocFields(BufferField::X | BufferField::GAMMA | BufferField::SIGMA | BufferField::INDEX);

        std::pair<unsigned int, unsigned int> numbers = initRoundJet(particleBuffer, boundaryBuffer, maxParticles);
        numParticles = numbers.first;
        unsigned int numBoundary = numbers.second;

        // Run VPM method
        runBoundaryVPM(
            maxParticles,
            numParticles,
            numBoundary,
            numTimeSteps,
            dt,
            numStepsVTK,
            uInf,
            particleBuffer,
            boundaryBuffer,
            new PedrizzettiRelaxation(0.3),
            new DynamicSFS(),
            KernelType::GAUSSIAN_ERF,
            blockSize,
            "test"
        );

        boundaryBuffer.freeFields();
        particleBuffer.freeFields();
        return;
    }
    }

    // Run VPM method
    runVPM(
        maxParticles,
        numParticles,
        numTimeSteps,
        dt,
        numStepsVTK,
        uInf,
        particleBuffer,
        new CorrectedPedrizzettiRelaxation(0.3),
        new DynamicSFS(),
        KernelType::WINCKELMAN,
        blockSize,
        "test"
    );

    particleBuffer.freeFields();
}