#include "main.hpp"
#include "vpmcore/kernel.h"
#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
    
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    //timeKernel(500);
    runSimulation();

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output the duration
    std::cout << "Kernel execution took " << milliseconds << " milliseconds." << std::endl;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}