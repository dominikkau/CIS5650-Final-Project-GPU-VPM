#include <vector>
#include <glm/glm.hpp>
#include <stdexcept>
#include <functional>
#include <particle.h>
#include <formulation.h>
#include <subfilterscale.h>
#include <viscousscheme.h>

// CHECK - Forward declarations
struct Relaxation;
struct FMM;
struct Kernel;
struct Particle;

// ParticleField definition
template <typename R=double, typename F=Formulation, typename V=ViscousScheme, typename S=SubFilterScale>
class ParticleField {
public:
    // User inputs
    int maxParticles;                      // Maximum number of particles
    std::vector<Particle> particles;       // Array of particles
    std::vector<void*> bodies;             // CHECK - Placeholder for ExaFMM array of bodies
    F formulation;                         // VPM formulation
    V viscous;                             // Viscous scheme

    // Internal properties
    int np;                            // Number of particles in the field
    int nt;                            // Current time step number
    R t;                          // Current time

    // Solver settings
    Kernel kernel;                        // Vortex particle kernel
    std::function<void()> UJ;              // Particle-to-particle calculation

    // Optional inputs
    std::function<glm::vec3(R)> Uinf;      // Uniform freestream function Uinf(t)
    S SFS;                                 // Subfilter-scale contributions scheme
    std::function<void()> integration;
    bool transposed;                       // Transposed vortex stretch scheme
    Relaxation relaxation;                // Relaxation scheme
    FMM fmm;                              // Fast-multipole settings

    // Internal memory for computation
    std::vector<R> M;

    // Constructor
    ParticleField(
        int maxparticles,
        std::vector<Particle> particles,
        td::vector<void*> bodies,
        F formulation,
        V viscous,
        int np = 0,
        int nt = 0,
        R t = R(0.0),
        Kernel kernel = kernel_default,
        std::function<void()> UJ=UJ_fmm,
        std::function<glm::vec3(R)> Uinf = Uinf_default,
        S SFS = SFS_default,
        sstd::function<void()> integration = rungekutta3,
        bool transposed = true,
        Relaxation relaxation = relaxation_default,
        Matrix M = M(4, 0.0))
    )
        : maxparticles(maxparticles),
          particles(particles),
          bodies(bodies),
          formulation(formulation),
          viscous(viscous),
          np(np),
          nt(nt),
          t(t),
          kernel(kernel),
          interaction(interaction),
          Uinf(Uinf),
          SFS(SFS),
          integration(integration),
          transposed(transposed),
          relaxation(relaxation),
          M(M) {}



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

template <typename T>
void _reset_particle(Particle& P, T tzero) {
    std::fill(P.U.begin(), P.U.end(), tzero);
    
    for (auto& row : P.J) {
        std::fill(row.begin(), row.end(), tzero);
    }

    std::fill(P.PSE.begin(), P.PSE.end(), tzero);
}

// Reset all particles in the ParticleField
template <typename R, typename F, typename V>
void _reset_particles(ParticleField<R, F, V>& field) {
    R tzero = R(0);
    for (auto& P : field.particles) {
        _reset_particle(P, tzero);
    }
}

// Function to reset the SFS (Subfilter Scale) data for an individual particle
template <typename T>
void _reset_particle_sfs(Particle & P, T tzero) {
    std::fill(P._SFS.begin(), P._SFS.end(), tzero);
}

// Reset SFS data for all particles in the ParticleField
template <typename R, typename F, typename V>
void _reset_particles_sfs(ParticleField<R, F, V>& field) {
    R tzero = R(0);  
    for (auto& P : field.particles) {
        _reset_particle_sfs(P, tzero);
    }
}

};
