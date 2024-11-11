#include <glm/glm.hpp>
#include <vector>

//CHECK - Initialization of member variables
#ifndef PARTICLE_H
#define PARTICLE_H

#include <glm/glm.hpp>

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

    // Constructor
    Particle();

    // Accessor functions
    glm::vec3 get_U() const;

    // Vorticity component calculations
    float get_W1() const;
    float get_W2() const;
    float get_W3() const;
    glm::vec3 get_W() const;

    // SFS component accessors and mutators
    float get_SFS1() const;
    float get_SFS2() const;
    float get_SFS3() const;
    void add_SFS1(float val);
    void add_SFS2(float val);
    void add_SFS3(float val);
};

#endif // PARTICLE_H
