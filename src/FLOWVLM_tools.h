#ifndef FLOWVLM_TOOLS_H
#define FLOWVLM_TOOLS_H

#include <vector>

// Declarations only (no implementations here)
std::vector<std::vector<double>> inverse(const std::vector<std::vector<double>>& matrix);
std::vector<double> countertransform(
    const std::vector<double>& Vp,
    const std::vector<std::vector<double>>& invM,
    const std::vector<double>& T);
std::vector<std::vector<double>> countertransformCollection(
    const std::vector<std::vector<double>>& Vps,
    const std::vector<std::vector<double>>& invM,
    const std::vector<double>& T);
double vectorNorm(const std::vector<double>& vec);
double dotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2);
bool checkCoordSys(const std::vector<std::vector<double>>& M, bool raise_error);

#endif // FLOWVLM_TOOLS_H
