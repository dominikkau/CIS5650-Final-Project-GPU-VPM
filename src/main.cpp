#include "main.hpp"
#include "vpmcore/kernel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {

    const std::string logFileName = "vortex_ring_simulation_log.txt";
    std::ofstream logFile(logFileName, std::ios::out);

    if (!logFile.is_open()) {
        std::cerr << "Failed to open log file: " << logFileName << std::endl;
        return 1;
    }

    // Log file header
    logFile << "Nphi,nc,Time(ms)" << std::endl;

    // Varying parameters
    std::vector<int> Nphis = { 100, 200, 300 }; // Add more values as needed
    std::vector<int> ncs = { 1, 2, 3 };         // Add more values as needed

    for (int Nphi : Nphis) {
        for (int nc : ncs) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, 0);

            try {
                runSimulation(Nphi, nc);
            }
            catch (const std::exception& e) {
                std::cerr << "Simulation failed for Nphi=" << Nphi
                    << ", nc=" << nc << ": " << e.what() << std::endl;
                logFile << Nphi << "," << nc << ",FAILED" << std::endl;
                continue; // Skip to the next combination
            }

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);

            // Output results to console and log file
            std::cout << "Nphi: " << Nphi << ", nc: " << nc
                << ", Time: " << milliseconds << " ms" << std::endl;

            logFile << Nphi << "," << nc << "," << milliseconds << std::endl;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    logFile.close();
    std::cout << "Results logged to " << logFileName << std::endl;

    return 0;
}