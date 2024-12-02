#include "particle.h"

// Constructor
Particle::Particle() 
    : X(0.0f), Gamma(0.0f), sigma(0.0f), vol(0.0f), circulation(0.0f), isStatic(false),
      U(0.0f), J(0.0f), PSE(0.0f), M(0.0f), C(0.0f), _SFS(0.0f) {}

    __host__ __device__ void Particle::reset(){
        U = glm::vec3(0);
        J = glm::mat3(0);
        PSE = glm::vec3(0);
    }
    
    __host__ __device__ void Particle::resetSFS(){
        SFS = glm::vec3(0);
    }
  