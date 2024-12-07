#include <vector>
#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "FLOWVLM_tools.h"


std::vector<std::vector<double>> inverse(const std::vector<std::vector<double>>& matrix) {
    // Convert std::vector to Eigen::Matrix
    size_t n = matrix.size();
    Eigen::MatrixXd eigenMatrix(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            eigenMatrix(i, j) = matrix[i][j];
        }
    }

    // Compute the inverse using Eigen
    Eigen::MatrixXd eigenInverse = eigenMatrix.inverse();

    // Convert Eigen::Matrix back to std::vector
    std::vector<std::vector<double>> result(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i][j] = eigenInverse(i, j);
        }
    }

    return result;
}

// Countertransform for a single vector
std::vector<double> countertransform(
    const std::vector<double>& Vp,
    const std::vector<std::vector<double>>& invM,
    const std::vector<double>& T) {

    if (Vp.size() != invM[0].size() || invM.size() != T.size()) {
        throw std::invalid_argument("Dimension mismatch in countertransform");
    }

    std::vector<double> result(T.size(), 0.0);

    // Matrix multiplication invM * Vp
    for (size_t i = 0; i < invM.size(); ++i) {
        for (size_t j = 0; j < Vp.size(); ++j) {
            result[i] += invM[i][j] * Vp[j];
        }
    }

    // Add translation vector T
    for (size_t i = 0; i < T.size(); ++i) {
        result[i] += T[i];
    }

    return result;
}

// Countertransform for a collection of vectors
std::vector<std::vector<double>> countertransformCollection(
    const std::vector<std::vector<double>>& Vps,
    const std::vector<std::vector<double>>& invM,
    const std::vector<double>& T) {

    std::vector<std::vector<double>> out;
    out.reserve(Vps.size());

    for (const auto& Vp : Vps) {
        out.push_back(countertransform(Vp, invM, T));
    }

    return out;
}


// Helper functions for vector operations
double vectorNorm(const std::vector<double>& vec) {
    double sum = 0.0;
    for (double val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

double dotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    double result = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// Check that the matrix M defines a coordinate system
bool checkCoordSys(const std::vector<std::vector<double>>& M, bool raise_error) {
    size_t dims = M.size();

    // Check normalization of vectors
    for (size_t i = 0; i < dims; ++i) {
        double norm = vectorNorm(M[i]);
        if (std::abs(norm - 1.0) > 1e-8) {
            if (raise_error) {
                throw std::runtime_error("Not unitary axis: vector " + std::to_string(i));
            }
            return false;
        }
    }

    // Check orthogonality of vectors
    for (size_t i = 0; i < dims; ++i) {
        const auto& xi = M[i];
        const auto& xip1 = M[(i + 1) % dims]; // Wrap-around for indexing
        double proj = std::abs(dotProduct(xi, xip1));
        if (proj > 1e-8) {
            if (raise_error) {
                throw std::runtime_error("Non-orthogonal system detected.");
            }
            return false;
        }
    }

    return true;
}

//// Overloaded function for higher abstraction
//bool checkCoordSys(const std::vector<std::vector<std::vector<double>>>& M_array, bool raise_error) {
//    size_t dims = 3; // Fixed size for 3D systems
//
//    // Check for automatic differentiation or type consistency
//    for (const auto& M : M_array) {
//        if (M.size() != dims || M[0].size() != dims) {
//            throw std::runtime_error("Invalid matrix dimensions for coordinate system check.");
//        }
//    }
//
//    for (const auto& M : M_array) {
//        if (!checkCoordSys(M, raise_error)) {
//            return false;
//        }
//    }
//
//    return true;
//}