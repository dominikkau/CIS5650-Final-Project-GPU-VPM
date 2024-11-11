#include <tuple>
#include <particle.h>

struct Particle {
    // User-defined variables
    glm::vec3 X;               // Position (3-element vector)
    glm::vec3 Gamma;           // Vectorial circulation (3-element vector)
    float sigma;               // Smoothing radius (scalar)
    float vol;                 // Volume (scalar)
    float circulation;         // Scalar circulation (scalar)
    bool isStatic;             // If true, this particle does not evolve in time

    // Properties
    glm::vec3 U;               // Velocity at particle (3-element vector)
    glm::mat3 J;               // Jacobian at particle (3x3 matrix)
    glm::vec3 PSE;             // Particle-strength exchange (3-element vector)

    // Internal variables
    glm::mat3 M;               // 3x3 matrix for auxiliary memory
    glm::vec3 C;               // C[0]=SFS coefficient, C[1]=numerator, C[2]=denominator
    glm::vec3 _SFS;            // 3x1 vector for SFS

    // Constructor with default initialization
    Particle() 
        : X(0.0f), Gamma(0.0f), sigma(0.0f), vol(0.0f), circulation(0.0f), isStatic(false),
          U(0.0f), J(0.0f), PSE(0.0f), M(0.0f), C(0.0f), _SFS(0.0f) {}

    // Functions for accessing properties
    glm::vec3 get_U() const { return U; }

    // Functions to compute vorticity components from the Jacobian
    float get_W1() const { return J[2][1] - J[1][2]; }
    float get_W2() const { return J[0][2] - J[2][0]; }
    float get_W3() const { return J[1][0] - J[0][1]; }

    // Returns all three components of the vorticity vector as a glm::vec3
    glm::vec3 get_W() const {
        return glm::vec3(get_W1(), get_W2(), get_W3());
    }

    // Functions for accessing and modifying SFS components
    float get_SFS1() const { return _SFS[0]; }
    float get_SFS2() const { return _SFS[1]; }
    float get_SFS3() const { return _SFS[2]; }

    void add_SFS1(float val) { _SFS[0] += val; }
    void add_SFS2(float val) { _SFS[1] += val; }
    void add_SFS3(float val) { _SFS[2] += val; }
};
