#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
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
    glm::vec3 SFS;               

    // Constructor
    Particle();

    // Reset particle Jacobian, PSE and SFS
    __host__ __device__ void Particle::reset();
    __host__ __device__ void Particle::resetSFS();
};