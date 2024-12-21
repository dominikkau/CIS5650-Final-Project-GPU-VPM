#include "main.hpp"
#include "vpmcore/kernel.h"
#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation();

    // Stop measuring time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Output the duration
    std::cout << "runSimulation took " << duration.count() << " seconds." << std::endl;

    return 0;
}