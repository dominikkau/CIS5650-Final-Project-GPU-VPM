#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <string>

//#define ENABLE_CUDA_ERROR
#define TRANSPOSED
//#define DOUBLE_PRECISION
//#define CLASSIC_VPM

#ifdef DOUBLE_PRECISION
    #define EPS 1e-9
    typedef glm::dvec3 vpmvec3;
    typedef glm::dvec2 vpmvec2;
    typedef glm::dmat3 vpmmat3;
    typedef double     vpmfloat;
    #define PI     3.14159265358979
    #define const1 0.06349363593424097
    #define const2 0.7978845608028654
    #define const3 0.238732414637843
    #define const4 0.07957747154594767
    #define sqrt2  1.4142135623730951
#else
    #define EPS 1e-6f
    typedef glm::fvec3 vpmvec3;
    typedef glm::fvec2 vpmvec2;
    typedef glm::fmat3 vpmmat3;
    typedef float      vpmfloat;
    #define PI     3.14159265358979f
    #define const1 0.06349363593424097f
    #define const2 0.7978845608028654f
    #define const3 0.238732414637843f
    #define const4 0.07957747154594767f
    #define sqrt2  1.4142135623730951f
#endif

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

__host__ __device__ inline vpmvec3 xDotNablaY(const vpmvec3& x, const vpmmat3& jacobianY) {
#ifdef TRANSPOSED
    return jacobianY * x;
#else
    return x * jacobianY;
#endif
}

__host__ __device__ inline vpmvec3 nablaCrossX(const vpmmat3& jacobianX) {
    return vpmvec3{
        jacobianX[1][2] - jacobianX[2][1],
        jacobianX[2][0] - jacobianX[0][2],
        jacobianX[0][1] - jacobianX[1][0]
    };
}

struct SingularKernel {
    __host__ __device__ inline vpmfloat zeta(vpmfloat r) { return (r == 0.0f) ? 1.0f : 0.0f; }
    __host__ __device__ inline vpmfloat g(vpmfloat r) { return 1.0f; }
    __host__ __device__ inline vpmfloat dgdr(vpmfloat r) { return 0.0f; }
    __host__ __device__ inline vpmvec2  g_dgdr(vpmfloat r) { return vpmvec2{ 1.0f, 0.0f }; }
};

struct GaussianKernel {
    __host__ __device__ inline vpmfloat zeta(vpmfloat r) {
        return const3 * exp(-r * r * r);
    }
    __host__ __device__ inline vpmfloat g(vpmfloat r) {
        return 1.0f - exp(-r * r * r);
    }
    __host__ __device__ inline vpmfloat dgdr(vpmfloat r) {
        return 3.0f * r * r * exp(-r * r * r);
    }
    __host__ __device__ inline vpmvec2 g_dgdr(vpmfloat r) {
        const vpmfloat tmp = exp(-r * r * r);
        return vpmvec2{ 1.0f - tmp, 3.0f * r * r * tmp };
    }
};

struct GaussianErfKernel {
    __host__ __device__ inline vpmfloat zeta(vpmfloat r) {
        return const1 * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline vpmfloat g(vpmfloat r) {
        return erf(r / sqrt2) - const2 * r * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline vpmfloat dgdr(vpmfloat r) {
        return const2 * r * r * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline vpmvec2  g_dgdr(vpmfloat r) {
        const vpmfloat tmp = const2 * r * exp(-r * r / 2.0f);
        return vpmvec2{ erf(r / sqrt2) - tmp, r * tmp };
    }
};

struct WinckelmansKernel {
    __host__ __device__ inline vpmfloat zeta(vpmfloat r) {
        return const4 * 7.5f / pow(r * r + 1.0f, 3.5f);
    }
    __host__ __device__ inline vpmfloat g(vpmfloat r) {
        return r * r * r * (r * r + 2.5f) / pow(r * r + 1.0f, 2.5f);
    }
    __host__ __device__ inline vpmfloat dgdr(vpmfloat r) {
        return 7.5f * r * r / pow(r * r + 1.0f, 3.5f);
    }
    __host__ __device__ inline vpmvec2  g_dgdr(vpmfloat r) {
        const vpmfloat tmp = pow(r * r + 1.0f, 2.5f);
        return vpmvec2{ r * r * r * (r * r + 2.5f) / tmp,
                          7.5f * r * r / (tmp * (r * r + 1.0f)) };
    }
};

template <typename R, typename S, typename K>
struct ParticleField;

struct Particle;
struct ParticleBuffer;

struct DynamicSFS {
    vpmfloat minC;
    vpmfloat maxC;
    vpmfloat alpha;
    vpmfloat relaxFactor; // relaxation factor for Lagrangian average
    bool forcePositive;

    DynamicSFS(vpmfloat minC = 0, vpmfloat maxC = 1, vpmfloat alpha = 0.667, vpmfloat relaxFactor = 0.005, bool forcePositive = true)
        : minC(minC), maxC(maxC), alpha(alpha), relaxFactor(relaxFactor), forcePositive(forcePositive) {}

    template <typename R, typename S, typename K>
    void operator()(ParticleField<R, S, K>& field, vpmfloat a, vpmfloat b, int numBlocks, int blockSize);
};

__global__ void calculateTemporary(int N, ParticleBuffer particles, bool testFilter);

__global__ void calculateCoefficient(int N, ParticleBuffer particles, vpmfloat zeta0,
    vpmfloat alpha, vpmfloat relaxFactor, bool forcePositive, vpmfloat minC, vpmfloat maxC);

struct NoSFS {
    template <typename R, typename S, typename K>
    void operator()(ParticleField<R, S, K>& field, vpmfloat a, vpmfloat b, int numBlocks, int blockSize);
};

struct PedrizzettiRelaxation {
    vpmfloat relaxFactor;
    PedrizzettiRelaxation(vpmfloat relaxFactor) : relaxFactor(relaxFactor) {}

    template <typename R, typename S, typename K>
    void operator()(int N, ParticleField<R, S, K>& field, int numBlocks, int blockSize);
};

__global__ void pedrizzettiRelax(int N, ParticleBuffer particles, vpmfloat relaxFactor);

struct CorrectedPedrizzettiRelaxation {
    vpmfloat relaxFactor;
    CorrectedPedrizzettiRelaxation(vpmfloat relaxFactor) : relaxFactor(relaxFactor) {}

    template <typename R, typename S, typename K>
    void operator()(int N, ParticleField<R, S, K>& field, int numBlocks, int blockSize);
};

__global__ void correctedPedrizzettiRelax(int N, ParticleBuffer particles, vpmfloat relaxFactor);

struct NoRelaxation {
    template <typename R, typename S, typename K>
    inline void operator()(int N, ParticleField<R, S, K>& field, int numBlocks, int blockSize) {}
};

struct Particle {
    vpmvec3 X;          // Position
    vpmvec3 Gamma;      // Vectorial circulation
    vpmfloat sigma;     // Smoothing radius
    int index;          // Indices of particles
    vpmvec3 U;          // Velocity at particle
    vpmmat3 J;          // Jacobian at particle
    vpmmat3 M;          // Auxiliary memory
    vpmvec3 C;          // SFS coefficient, numerator, denominator
    vpmvec3 SFS;

    /*vpmfloat vol;            // Volume
    vpmfloat circulation;    // Scalar circulation
    bool isStatic;           // Indicates if particle is static
    vpmvec3 PSE;             // Particle-strength exchange*/

    // Constructor
    Particle()
        : X(0.0f), Gamma(0.0f), sigma(0.0f), 
        U(0.0f), J(0.0f),  M(0.0f), C(0.0f), SFS(0.0f), index(0) {}
        //PSE(0.0f), vol(0.0f), circulation(0.0f), isStatic(false), 
 
    __host__ __device__ void Particle::reset();    // Reset particle U, J and PSE
    __host__ __device__ void Particle::resetSFS(); // Reset particle SFS
};

enum BufferType {
	BUFFER_NONE = 0,
    BUFFER_X = 1 << 0,
    BUFFER_U = 1 << 1,
    BUFFER_J = 1 << 2,
    BUFFER_GAMMA = 1 << 3,
    BUFFER_SIGMA = 1 << 4,
    BUFFER_SFS = 1 << 5,
    BUFFER_C = 1 << 6,
    BUFFER_M = 1 << 7,
    BUFFER_INDEX = 1 << 8,
	BUFFER_PSE = 1 << 9,
	BUFFER_VOL = 1 << 10,
	BUFFER_CIRCULATION = 1 << 11,
	BUFFER_ISSTATIC = 1 << 12,
	BUFFER_ALL = 0xFFFF
    // Add other buffers as needed
};

enum OutputType {
    OUTPUT_NONE = 0,
    OUTPUT_X = 1 << 0,
    OUTPUT_U = 1 << 1,
    OUTPUT_OMEGA = 1 << 2,
    OUTPUT_GAMMA = 1 << 3,
    OUTPUT_SIGMA = 1 << 4,
    OUTPUT_INDEX = 1 << 5,
    OUTPUT_ALL = 0xFFFF
};

// ParticleField definition
template <typename R = PedrizzettiRelaxation, typename S = NoSFS, typename K = GaussianErfKernel>
struct ParticleField {
    unsigned int maxParticles;           // Maximum number of particles
    ParticleBuffer particles;        // Pointer to host particle buffer
    ParticleBuffer dev_particles;    // Pointer to device particle buffer
    unsigned int numParticles;           // Number of particles in the field
    unsigned int timeStep;               // Current time step
    K kernel;                   // Vortex particle kernel
    vpmvec3 uInf;               // Uniform freestream function
    S sfs;                      // Subfilter-scale contributions scheme
    R relaxation;               // Relaxation scheme
    int synchronized;           // Flags if host buffers are synchronized with device

    // Constructor
    ParticleField(
        unsigned int maxParticles,
        ParticleBuffer particles,
        unsigned int numParticles,
        int bufferMask,
        unsigned int timeStep = 0,
        K kernel = GaussianErfKernel(),
        vpmvec3 uInf = vpmvec3(0, 0, 0),
        S sfs = NoSFS(),
        R relaxation = PedrizzettiRelaxation(0.005)
    );
    // Destructor
    ~ParticleField();

    void initParticlesDevice();

	void syncParticlesDeviceToHost(int bufferMask);
	void syncParticlesHostToDevice(int bufferMask);

    void addParticleDevice(Particle& particle);
    void overwriteParticleDevice(Particle& particle, unsigned int index);
    void removeParticleDevice(unsigned int index);
    void cpyParticlesDeviceToDevice(ParticleBuffer inParticles, unsigned int inNumParticles,
        unsigned int startIndex, int bufferMask);
};

struct ParticleBuffer {
    vpmvec3* X = NULL;          // Position
    vpmvec3* Gamma = NULL;      // Vectorial circulation
    vpmfloat* sigma = NULL;     // Smoothing radius
    int* index = NULL;          // Indices of particles
    vpmvec3* U = NULL;          // Velocity at particle
    vpmmat3* J = NULL;          // Jacobian at particle
    vpmmat3* M = NULL;          // Auxiliary memory
    vpmvec3* C = NULL;          // SFS coefficient, numerator, denominator
    vpmvec3* SFS = NULL;

    /*vpmfloat* vol;            // Volume
    vpmfloat* circulation;    // Scalar circulation
    bool* isStatic;           // Indicates if particle is static
    vpmvec3* PSE;             // Particle-strength exchange*/
};

unsigned int cpyParticleBuffer(ParticleBuffer destBuffer, ParticleBuffer srcBuffer, unsigned int destNumParticles,
    unsigned int destMaxParticles, unsigned int srcNumParticles, unsigned int destIndex, int bufferMask, cudaMemcpyKind cpyDirection);

__global__ void resetParticles(int N, ParticleBuffer particles);
__global__ void resetParticlesSFS(int N, ParticleBuffer particles);

template <typename K>
__global__ void calcEstrNaive(int targetN, int sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, K kernel, bool reset=false, vpmfloat testFilterFactor=1.0);

template <typename K>
__global__ void calcVelJacNaive(int targetN, int sourceN, ParticleBuffer targetParticles,
    ParticleBuffer sourceParticles, K kernel, bool reset=false, vpmfloat testFilterFactor=1.0);

__global__ void rungeKuttaStep(int N, ParticleBuffer particles, vpmfloat a, vpmfloat b, vpmfloat dt,
    vpmfloat zeta0, vpmvec3 Uinf);

template <typename R, typename S, typename K>
void rungeKutta(ParticleField<R, S, K>& field, vpmfloat dt, bool useRelax, int numBlocks, int blockSize);

void runSimulation();

template <typename R, typename S, typename K>
void runVPM(
    unsigned int maxParticles,
    unsigned int numParticles,
    unsigned int numTimeSteps,
    vpmfloat dt,
    unsigned int fileSaveSteps,
    vpmvec3 uInf,
    ParticleBuffer particleBuffer,
    R relaxation,
    S sfs,
    K kernel,
    int blockSize,
    std::string filename
);

//void timeKernel(int repetitions);