#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <string>

#define TRANSPOSED
//#define SHARED_MEMORY
#define DOUBLE_PRECISION
#define BLOCK_SIZE 128

#define PI 3.14159265358979f
#define const1 0.06349363593424097f
#define const2 0.7978845608028654f
#define const3 0.238732414637843f
#define const4 0.07957747154594767f
#define sqrt2 1.4142135623730951f

#ifdef DOUBLE_PRECISION
#define EPS 1e-9
typedef glm::dvec3 vpmvec3;
typedef glm::dvec2 vpmvec2;
typedef glm::dmat3 vpmmat3;
typedef double     vpmfloat;
#else
#define EPS 1e-6f
typedef glm::fvec3 vpmvec3;
typedef glm::fvec2 vpmvec2;
typedef glm::fmat3 vpmmat3;
typedef float      vpmfloat;
#endif

__host__ __device__ inline vpmvec3 xDotNablaY(vpmvec3 x, vpmmat3 jacobianY) {
#ifdef TRANSPOSED
    return jacobianY * x;
#else
    return x * jacobianY;
#endif
}

__host__ __device__ inline vpmvec3 nablaCrossX(vpmmat3 jacobianX) {
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
    __host__ __device__ inline vpmvec2 g_dgdr(vpmfloat r) { return vpmvec2{ 1.0f, 0.0f }; }
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
        vpmfloat tmp = exp(-r * r * r);
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
    __host__ __device__ inline vpmvec2 g_dgdr(vpmfloat r) {
        vpmfloat tmp = const2 * r * exp(-r * r / 2.0f);
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
    __host__ __device__ inline vpmvec2 g_dgdr(vpmfloat r) {
        vpmfloat tmp = pow(r * r + 1.0f, 2.5f);
        return vpmvec2{ r * r * r * (r * r + 2.5f) / tmp,
                          7.5f * r * r / (tmp * (r * r + 1.0f)) };
    }
};

template <typename R, typename S, typename K>
struct ParticleField;

struct Particle;

template <typename Rs, typename Ss, typename Ks, typename Rt, typename St, typename Kt, typename K>
__device__ void calcVelJacNaive(int index, ParticleField<Rs, Ss, Ks>* source, ParticleField<Rt, St, Kt>* target, K kernel);

template <typename R, typename S, typename K>
__device__ void calcVelJacNaive(int index, ParticleField<R, S, K>* field);

template <typename R, typename S, typename K>
__global__ void rungekutta(int N, ParticleField<R, S, K>* field, vpmfloat dt, bool relax);

template <typename Rs, typename Ss, typename Ks, typename Rt, typename St, typename Kt, typename K>
__device__ void calcEstrNaive(int index, ParticleField<Rs, Ss, Ks>* source, ParticleField<Rt, St, Kt>* target, K kernel);
template <typename R, typename S, typename K>
__device__ void calcEstrNaive(int index, ParticleField<R, S, K>* field);
template <typename R, typename S, typename K>
__device__ void dynamicProcedure(int index, ParticleField<R, S, K>* field, vpmfloat alpha, vpmfloat relaxFactor,
                                 bool forcePositive, vpmfloat minC, vpmfloat maxC);

struct DynamicSFS {
    vpmfloat minC;
    vpmfloat maxC;
    vpmfloat alpha;
    vpmfloat relaxFactor;
    bool forcePositive;

    DynamicSFS(vpmfloat minC = 0, vpmfloat maxC = 1, vpmfloat alpha = 0.667, vpmfloat relaxFactor = 0.005, bool forcePositive = true)
        : minC(minC), maxC(maxC), alpha(alpha), relaxFactor(relaxFactor), forcePositive(forcePositive) {}

    template <typename R, typename S, typename K>
    __device__ void operator()(int index, ParticleField<R, S, K>* field, vpmfloat a, vpmfloat b);
};

struct NoSFS {
    template <typename R, typename S, typename K>
    __device__ void operator()(int index, ParticleField<R, S, K>* field, vpmfloat a, vpmfloat b);
};

struct PedrizzettiRelaxation {
    vpmfloat relaxFactor;
    PedrizzettiRelaxation(vpmfloat relaxFactor) : relaxFactor(relaxFactor) {}
    template <typename R, typename S, typename K>
    __global__ void operator()(int N, ParticleField<R, S, K>* field);
};

struct CorrectedPedrizzettiRelaxation {
    vpmfloat relaxFactor;
    CorrectedPedrizzettiRelaxation(vpmfloat relaxFactor) : relaxFactor(relaxFactor) {}
    template <typename R, typename S, typename K>
    __global__ void operator()(int N, ParticleField<R, S, K>* field);
};

struct NoRelaxation {
    template <typename R, typename S, typename K>
    __global__ void operator()(int N, ParticleField<R, S, K>* field);
};

// ParticleField definition
template <typename R=PedrizzettiRelaxation, typename S=NoSFS, typename K=GaussianErfKernel>
class ParticleField {
public:
    // User inputs
    int maxParticles;                      // Maximum number of particles
    Particle* particles;                // Pointer to particle buffer
    // std::vector<void*> bodies;             // CHECK - Placeholder for ExaFMM array of bodies
    // F formulation;                         // VPM formulation
    // V viscous;                             // Viscous scheme

    // Internal properties
    int np;                            // Number of particles in the field
    int nt;                            // Current time step number
    vpmfloat t;                          // Current time

    // Solver settings
    K kernel;                        // Vortex particle kernel
    // std::function<void()> UJ;              // Particle-to-particle calculation

    // Optional inputs
    vpmvec3 Uinf;             // Uniform freestream function Uinf(t)
    S SFS;                      // Subfilter-scale contributions scheme
    bool transposed;            // Transposed vortex stretch scheme
    R relaxation;             // Relaxation scheme
    // FMM fmm;                 // Fast-multipole settings


    // Constructor
    ParticleField(
        int maxparticles,
        // std::vector<void*> bodies,
        Particle* particles,
        int np = 0,
        int nt = 0,
        vpmfloat t = 0.0f,
        K kernel = GaussianErfKernel(),
        // std::function<void()> UJ=UJ_fmm,
        vpmvec3 Uinf = vpmvec3(0, 0, 0),
        S SFS = NoSFS(),
        bool transposed = true,
        R relaxation = PedrizzettiRelaxation(0.005))
        :
        //   particles(particles),
        //   bodies(bodies),
        //   formulation(formulation),
        //   viscous(viscous),
        maxParticles(maxparticles),
        particles(particles),
        np(np),
        nt(nt),
        t(t),
        kernel(kernel),
        Uinf(Uinf),
        SFS(SFS),
        transposed(transposed),
        relaxation(relaxation) {};
};

struct Particle {
    // User-defined variables
    vpmvec3 X;               // Position
    vpmvec3 Gamma;           // Vectorial circulation
    vpmfloat sigma;               // Smoothing radius
    vpmfloat vol;                 // Volume
    vpmfloat circulation;         // Scalar circulation
    bool isStatic;             // Indicates if particle is static

    // Properties
    vpmvec3 U;               // Velocity at particle
    vpmmat3 J;               // Jacobian at particle
    vpmvec3 PSE;             // Particle-strength exchange

    // Internal variables
    vpmmat3 M;               // Auxiliary memory
    vpmvec3 C;               // SFS coefficient, numerator, denominator
    vpmvec3 SFS;               

    // Constructor
    Particle();
 
    __host__ __device__ void Particle::reset(); // Reset particle U, J and PSE
    __host__ __device__ void Particle::resetSFS(); // Reset particle SFS
};

void randomCubeInit(Particle* particleBuffer, int N, vpmfloat cubeSize = 10.0f, vpmfloat maxCirculation = 1.0f, vpmfloat maxSigma = 1.0f);
void randomSphereInit(Particle* particleBuffer, int N, vpmfloat sphereRadius = 10.0f, vpmfloat maxCirculation = 1.0f, vpmfloat maxSigma = 1.0f);
void runSimulation();

template <typename R, typename S, typename K>
void runVPM(
    int maxParticles,
    int numParticles,
    int numTimeSteps,
    vpmfloat dt,
    int fileSaveSteps,
    int blockSize,
    vpmvec3 uInf,
    Particle* particleBuffer,
    R relaxation,
    S sfs,
    K kernel,
    std::string filename
);