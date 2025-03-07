#pragma once

#include "common.h"

enum BufferField {
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

enum ParticleBufferType {
    BUFFER_DEVICE,
    BUFFER_HOST,
    BUFFER_HOST_PINNED
};

struct ParticleBuffer {
    const ParticleBufferType bufferType;
    int bufferFields = 0;
    vpmvec3* X = NULL;          // Position
    vpmvec3* Gamma = NULL;      // Vectorial circulation
    vpmfloat* sigma = NULL;     // Smoothing radius
    int* index = NULL;          // Indices of particles
    vpmvec3* U = NULL;          // Velocity at particle
    vpmmat3* J = NULL;          // Jacobian at particle
    vpmmat3* M = NULL;          // Auxiliary memory
    vpmvec3* C = NULL;          // SFS coefficient, numerator, denominator
    vpmvec3* SFS = NULL;

    /*vpmfloat* vol = NULL;           // Volume
    vpmfloat* circulation = NULL;   // Scalar circulation
    bool* isStatic = NULL;          // Indicates if particle is static
    vpmvec3* PSE = NULL;            // Particle-strength exchange*/

    ParticleBuffer(ParticleBufferType bufferType) : bufferType(bufferType) {};
    ~ParticleBuffer() { freeFields(bufferFields); };

    void mallocFields(unsigned int numParticles, int bufferMask);
    void freeFields(int bufferMask);
};

void _cpyParticleBuffer(ParticleBuffer destBuffer, ParticleBuffer srcBuffer,
    unsigned int destIndex, unsigned int srcNumParticles, int bufferMask);
unsigned int cpyParticleBuffer(ParticleBuffer destBuffer, ParticleBuffer srcBuffer, unsigned int destNumParticles,
    unsigned int destMaxParticles, unsigned int srcNumParticles, unsigned int destIndex, int bufferMask);