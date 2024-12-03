#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#define TRANSPOSED

#define EPS 1e-6f
#define PI 3.14159265358979f
#define const1 0.06349363593424097f
#define const2 0.7978845608028654f
#define const3 0.238732414637843f
#define const4 0.07957747154594767f
#define sqrt2 1.4142135623730951f

__host__ __device__ inline glm::vec3 xDotNablaY(glm::vec3 x, glm::mat3 jacobianY) {
#ifdef TRANSPOSED
    return jacobianY * x;
#else
    return x * jacobianY;
#endif
}

__host__ __device__ inline glm::vec3 nablaCrossX(glm::mat3 jacobianX) {
    return glm::vec3{
        jacobianX[1][2] - jacobianX[2][1],
        jacobianX[2][0] - jacobianX[0][2],
        jacobianX[0][1] - jacobianX[1][0]
    };
}

struct SingularKernel {
    __host__ __device__ inline float zeta(float r) { return (r == 0.0f) ? 1.0f : 0.0f; }
    __host__ __device__ inline float g(float r) { return 1.0f; }
    __host__ __device__ inline float dgdr(float r) { return 0.0f; }
    __host__ __device__ inline glm::vec2 g_dgdr(float r) { return glm::vec2{ 1.0f, 0.0f }; }
};

struct GaussianKernel {
    __host__ __device__ inline float zeta(float r) {
        return const3 * exp(-r * r * r);
    }
    __host__ __device__ inline float g(float r) {
        return 1.0f - exp(-r * r * r);
    }
    __host__ __device__ inline float dgdr(float r) {
        return 3.0f * r * r * exp(-r * r * r);
    }
    __host__ __device__ inline glm::vec2 g_dgdr(float r) {
        float tmp = exp(-r * r * r);
        return glm::vec2{ 1.0f - tmp, 3.0f * r * r * tmp };
    }
};

struct GaussianErfKernel {
    __host__ __device__ inline float zeta(float r) {
        return const1 * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline float g(float r) {
        return erf(r / sqrt2) - const2 * r * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline float dgdr(float r) {
        return const2 * r * r * exp(-r * r / 2.0f);
    }
    __host__ __device__ inline glm::vec2 g_dgdr(float r) {
        float tmp = const2 * r * exp(-r * r / 2.0f);
        return glm::vec2{ erf(r / sqrt2) - tmp, r * tmp };
    }
};

struct WinckelmansKernel {
    __host__ __device__ inline float zeta(float r) {
        return const4 * 7.5f / pow(r * r + 1.0f, 3.5f);
    }
    __host__ __device__ inline float g(float r) {
        return r * r * r * (r * r + 2.5f) / pow(r * r + 1.0f, 2.5f);
    }
    __host__ __device__ inline float dgdr(float r) {
        return 7.5f * r * r / pow(r * r + 1.0f, 3.5f);
    }
    __host__ __device__ inline glm::vec2 g_dgdr(float r) {
        float tmp = pow(r * r + 1.0f, 2.5f);
        return glm::vec2{ r * r * r * (r * r + 2.5f) / tmp,
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
__global__ void rungekutta(int N, ParticleField<R, S, K>* field, float dt, bool relax);

template <typename Rs, typename Ss, typename Ks, typename Rt, typename St, typename Kt, typename K>
__device__ void calcEstrNaive(int index, ParticleField<Rs, Ss, Ks>* source, ParticleField<Rt, St, Kt>* target, K kernel);
template <typename R, typename S, typename K>
__device__ void calcEstrNaive(int index, ParticleField<R, S, K>* field);
template <typename R, typename S, typename K>
__device__ void dynamicProcedure(int index, ParticleField<R, S, K>* field, float alpha, float relaxFactor,
                                 bool forcePositive, float minC, float maxC);

struct DynamicSFS {
    float minC;
    float maxC;
    float alpha;
    float relaxFactor;
    bool forcePositive;

    DynamicSFS(float minC = 0, float maxC = 1, float alpha = 0.667, float relaxFactor = 0.005, bool forcePositive = true)
        : minC(minC), maxC(maxC), alpha(alpha), relaxFactor(relaxFactor), forcePositive(forcePositive) {}

    template <typename R, typename S, typename K>
    __device__ void operator()(int index, ParticleField<R, S, K>* field, float a, float b);
};

struct NoSFS {
    template <typename R, typename S, typename K>
    __device__ void operator()(int index, ParticleField<R, S, K>* field, float a, float b);
};

struct PedrizzettiRelaxation {
    float relaxFactor;
    PedrizzettiRelaxation(float relaxFactor) : relaxFactor(relaxFactor) {}
    __device__ void operator()(Particle& particle);
};

struct CorrectedPedrizzettiRelaxation {
    float relaxFactor;
    CorrectedPedrizzettiRelaxation(float relaxFactor) : relaxFactor(relaxFactor) {}
    __device__ void operator()(Particle& particle);
};

struct NoRelaxation {
    __device__ void operator()(Particle& particle);
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
    float t;                          // Current time

    // Solver settings
    K kernel;                        // Vortex particle kernel
    // std::function<void()> UJ;              // Particle-to-particle calculation

    // Optional inputs
    glm::vec3 Uinf;             // Uniform freestream function Uinf(t)
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
        float t = 0.0f,
        K kernel = GaussianErfKernel(),
        // std::function<void()> UJ=UJ_fmm,
        glm::vec3 Uinf = glm::vec3(0, 0, 0),
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
    glm::vec3 X;               // Position
    glm::vec3 Gamma;           // Vectorial circulation
    float sigma;               // Smoothing radius
    float vol;                 // Volume
    float circulation;         // Scalar circulation
    bool isStatic;             // Indicates if particle is static

    // Properties
    glm::vec3 U;               // Velocity at particle
    glm::mat3 J;               // Jacobian at particle
    glm::vec3 PSE;             // Particle-strength exchange

    // Internal variables
    glm::mat3 M;               // Auxiliary memory
    glm::vec3 C;               // SFS coefficient, numerator, denominator
    glm::vec3 SFS;               

    // Constructor
    Particle();

    // Reset particle Jacobian, PSE and SFS
    __host__ __device__ void Particle::reset();
    __host__ __device__ void Particle::resetSFS();
};

void randomCubeInit(Particle* particleBuffer, int N, float cubeSize = 10.0f, float maxCirculation = 1.0f, float maxSigma = 1.0f);
void randomSphereInit(Particle* particleBuffer, int N, float sphereRadius = 10.0f, float maxCirculation = 1.0f, float maxSigma = 1.0f);
void runSimulation();

template <typename R, typename S, typename K>
void runVPM(
    int maxParticles,
    int numParticles,
    int numTimeSteps,
    float dt,
    int fileSaveSteps,
    int blockSize,
    glm::vec3 uInf,
    Particle* particleBuffer,
    R relaxation,
    S sfs,
    K kernel
);