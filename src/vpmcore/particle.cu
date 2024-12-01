#include "particle.h"

// Constructor
Particle::Particle() 
    : X(0.0f), Gamma(0.0f), sigma(0.0f), vol(0.0f), circulation(0.0f), isStatic(false),
      U(0.0f), J(0.0f), PSE(0.0f), M(0.0f), C(0.0f), _SFS(0.0f) {}

glm::vec3 get_U(const Particle& P) {
    return P.U;
}

std::tuple<float, float, float> get_W(const Particle& P) {
    return {get_W1(P), get_W2(P), get_W3(P)};
}

// Vorticity component calculations
float get_W1(const Particle& P) {
    return P.J[2][1] - P.J[1][2];
}

float get_W2(const Particle& P) {
    return P.J[0][2] - P.J[2][0];
}

float get_W3(const Particle& P) {
    return P.J[1][0] - P.J[0][1];
}

// SFS component getters and mutators
float get_SFS1(const Particle& P) {
    return P.C[0];
}

float get_SFS2(const Particle& P) {
    return P.C[1];
}

float get_SFS3(const Particle& P) {
    return P.C[2];
}

void add_SFS1(Particle& P, float val) {
    P.C[0] += val;
}

void add_SFS2(Particle& P, float val) {
    P.C[1] += val;
}

void add_SFS3(Particle& P, float val) {
    P.C[2] += val;
}