#include "main.hpp"
#include <cuda_runtime.h>
#include "vpmcore/particle.h"
#include "vpmcore/timeIntegration.h"

int main(int argc, char* argv[]) {
    runVPM();

    return 0;
}

void runVPM() {
    int numParticles{ 1000 };

    int blockSize{ 128 };
    int fullBlocksPerGrid{ (numParticles + blockSize - 1) / blockSize };
    
    // Declare host particle buffer
    Particle* particleBuffer = new Particle[numParticles];
    // Declare device particle buffer
    Particle* dev_particleBuffer;
    cudaMalloc((void**)&dev_particleBuffer, numParticles * sizeof(Particle));

    // Initialize host Buffer somehow

    // Copy particle buffer from host to device
    cudaMemcpy(dev_particleBuffer, particleBuffer, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    // rungekutta<<< >>>();
}