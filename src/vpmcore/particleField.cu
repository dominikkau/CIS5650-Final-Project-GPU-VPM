#include <vector>
#include <glm/glm.hpp>
#include <stdexcept>
#include <functional>
#include <particle.h>
#include <subFilterScale.h>
#include <relaxation.h>
#include <kernels.h>

// ParticleField definition
template <typename R=double, typename Rel=PedrizzettiRelaxation, typename S=DynamicSFS, typename K = GaussianErfKernel>
class ParticleField {
public:
    // User inputs
    int maxParticles;                      // Maximum number of particles
    // std::vector<void*> bodies;             // CHECK - Placeholder for ExaFMM array of bodies
    // F formulation;                         // VPM formulation
    // V viscous;                             // Viscous scheme

    // Internal properties
    int np;                            // Number of particles in the field
    int nt;                            // Current time step number
    R t;                          // Current time

    // Solver settings
    K kernel;                        // Vortex particle kernel
    // std::function<void()> UJ;              // Particle-to-particle calculation

    // Optional inputs
    glm::vec3 Uinf;      // Uniform freestream function Uinf(t)
    S SFS;                                 // Subfilter-scale contributions scheme
    bool transposed;                       // Transposed vortex stretch scheme
    Rel relaxation;                // Relaxation scheme
    FMM fmm;                              // Fast-multipole settings


    // Constructor
    ParticleField(
        int maxparticles,
        // std::vector<void*> bodies,
        int np = 0,
        int nt = 0,
        R t = R(0.0),
        K kernel = GaussianErfKernel(),
        // std::function<void()> UJ=UJ_fmm,
        glm::vec3 Uinf = glm::vec3(0, 0, 0),
        S SFS = DynamicSFS(),
        bool transposed = true,
        Rel relaxation = PedrizzettiRelaxation(0.005))
        : 
        //   particles(particles),
        //   bodies(bodies),
        //   formulation(formulation),
        //   viscous(viscous),
          maxparticles(maxparticles),
          np(np),
          nt(nt),
          t(t),
          kernel(kernel),
          Uinf(Uinf),
          SFS(SFS),
          transposed(transposed),
          relaxation(relaxation), {}



    // Methods
    bool isLES() const {
        return SFS.isEnabled();
    }

    void addParticle(const glm::vec3& X, const glm::vec3& Gamma, const glm::vec3& sigma, 
                     R vol = R(0), R circulation = R(1), 
                     R C = R(0), bool isStatic = false, int index = -1) {
        if (np == maxParticles) {
            throw std::runtime_error("PARTICLE OVERFLOW: Max number of particles reached.");
        }

        auto& P = particles[np];
        P.X = X;
        P.Gamma = Gamma;
        P.sigma = sigma;
        P.vol = vol;
        P.circulation = std::abs(circulation);
        P.C = C;
        P.isStatic = isStatic;
        P.index = (index == -1) ? np + 1 : index;

        ++np;
    }

    void addParticle(const Particle& P) {
        addParticle(P.X, P.Gamma, P.sigma, P.vol, P.circulation, P.C, P.isStatic);
    }

    int getNp() const {
        return np;
    }

    Particle& getParticle(int i, bool emptyParticle = false) {
        if (i <= 0) {
            throw std::runtime_error("Invalid particle index.");
        }
        if (!emptyParticle && i > np) {
            throw std::runtime_error("Particle index exceeds current particle count.");
        }
        if (emptyParticle && i != (np + 1)) {
            throw std::runtime_error("Requested empty particle index is not valid.");
        }
        return particles[i - 1];
    }

    void removeParticle(int i) {
        if (i <= 0 || i > np) {
            throw std::runtime_error("Invalid particle index for removal.");
        }

        if (i != np) {
            particles[i - 1] = particles[np - 1];
        }
        --np;
    }

    void nextStep(R dt) {
        if (np > 0 && integration) {
            integration(*this, dt);
        }
        t += dt;
        ++nt;
    }

    void resetParticles() {
        for (auto& P : particles) {
            P.reset();
        }
    }
};

