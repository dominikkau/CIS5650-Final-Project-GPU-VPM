#include "constants.h"
#include <functional>
#include "FLOWVLM_solver.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <optional>
#include <numeric>

// Forward declaration of the regularize and smoothing functions
bool regularize = true; // Set to true if regularization is needed
double core_rad = 1.0; // Core radius for regularization
double col_crit = 1e-6; // Collinearity threshold
bool mute_warning = false; // Mute warnings
bool blobify = false; // Flag for blobification

using namespace VLMSolver;
using namespace std;

// ------------ PARAMETERS ------------------------------------------------------
// Criteria for collinearity
// NOTE: Anything less than 1 / 10 ^ 15 reaches float precision.
const double col_crit = 1e-8;

int n_col = 0;  // Number of colinears found
bool mute_warning = false;
bool regularize = false;
bool blobify = false;
double smoothing_rad = 1e-9;

void VLMSolver::_mute_warning(bool booln) {
    mute_warning = booln;
}

void VLMSolver::_regularize(bool booln) {
    regularize = booln;
}

void VLMSolver::_blobify(bool booln) {
    blobify = booln;
}

void VLMSolver::_smoothing_rad(double val) {
    smoothing_rad = val;
}

// Wincklman's regularizing function
double VLMSolver::gw(double r, double sgm) {
    double ratio = r / sgm;
    return std::pow(ratio, 3) * (std::pow(ratio, 2) + 2.5) / std::pow((std::pow(ratio, 2) + 1), 2.5);
}

// ------------ HELPER FUNCTIONS -----------------------------------------------
bool VLMSolver::check_collinear(double magsqr, double col_crit, bool ign_col) {
    if (magsqr < col_crit || std::isnan(magsqr)) {
        if (!ign_col) {
            if (n_col == 0 && !mute_warning) {
                cout << "Requested induced velocity on a point colinear with vortex. Returning 0" << endl;
            }
        }
        return true;
    }
    return false;
}

// ------------ VORTEX FUNCTIONS ----------------------------------------------
// Function to compute the Euclidean norm of a 3D vector
double norm(const std::vector<double>& v) {
    return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
}

// Calculates the induced velocity of the bound vortex AB on point C
std::vector<double> _V_AB(const std::vector<double>& A, const std::vector<double>& B, const std::vector<double>& C,
    double gamma, bool ign_col = false) {
    std::vector<double> r0 = { B[0] - A[0], B[1] - A[1], B[2] - A[2] };
    std::vector<double> r1 = { C[0] - A[0], C[1] - A[1], C[2] - A[2] };
    std::vector<double> r2 = { C[0] - B[0], C[1] - B[1], C[2] - B[2] };

    std::vector<double> crss = { r1[1] * r2[2] - r1[2] * r2[1],
                                 r1[2] * r2[0] - r1[0] * r2[2],
                                 r1[0] * r2[1] - r1[1] * r2[0] };

    double magsqr = std::inner_product(crss.begin(), crss.end(), crss.begin(), 0.0) + (regularize ? core_rad : 0);

    // Checks colinearity
    if (check_collinear(magsqr / norm(r0), col_crit, ign_col)) {
        if (!ign_col && n_col == 1 && !mute_warning) {
            /*std::cout << "\n\t magsqr:" << magsqr << " \n\t A:(" << A[0] << "," << A[1] << "," << A[2] << ")"
                << " \n\t B:(" << B[0] << "," << B[1] << "," << B[2] << ")"
                << " \n\t C:(" << C[0] << "," << C[1] << "," << C[2] << ")" << std::endl;*/
        }
        return { 0.0, 0.0, 0.0 };
    }

    std::vector<double> F1 = { crss[0] / magsqr, crss[1] / magsqr, crss[2] / magsqr };
    std::vector<double> aux = { r1[0] / sqrt(std::inner_product(r1.begin(), r1.end(), r1.begin(), 0.0)) -
                                r2[0] / sqrt(std::inner_product(r2.begin(), r2.end(), r2.begin(), 0.0)),
                                r1[1] / sqrt(std::inner_product(r1.begin(), r1.end(), r1.begin(), 0.0)) -
                                r2[1] / sqrt(std::inner_product(r2.begin(), r2.end(), r2.begin(), 0.0)),
                                r1[2] / sqrt(std::inner_product(r1.begin(), r1.end(), r1.begin(), 0.0)) -
                                r2[2] / sqrt(std::inner_product(r2.begin(), r2.end(), r2.begin(), 0.0)) };
    double F2 = std::inner_product(r0.begin(), r0.end(), aux.begin(), 0.0);

    if (blobify) {
        // std::cout << "Blobified! " << smoothing_rad << std::endl;
        F1[0] *= gw(norm(crss) / norm(r0), smoothing_rad);
        F1[1] *= gw(norm(crss) / norm(r0), smoothing_rad);
        F1[2] *= gw(norm(crss) / norm(r0), smoothing_rad);
    }

    if (gamma == 0.0) {
        return { F1[0] * F2, F1[1] * F2, F1[2] * F2 };
    }
    else {
        return { (gamma / (4 * M_PI)) * F1[0] * F2,
                (gamma / (4 * M_PI)) * F1[1] * F2,
                (gamma / (4 * M_PI)) * F1[2] * F2 };
    }
}
// Dot product of two 3D vectors
double dot(const std::vector<double>& v1, const std::vector<double>& v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// Magnitude squared of a 3D vector
double magsqr(const std::vector<double>& v) {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

std::vector<double> _V_Ainf_out(const std::vector<double>& A, const std::vector<double>& infD,
    const std::vector<double>& C, double gamma, bool ign_col = false) {
    std::vector<double> AC = { C[0] - A[0], C[1] - A[1], C[2] - A[2] };

    std::vector<double> unitinfD = { infD[0] / sqrt(std::inner_product(infD.begin(), infD.end(), infD.begin(), 0.0)),
                                     infD[1] / sqrt(std::inner_product(infD.begin(), infD.end(), infD.begin(), 0.0)),
                                     infD[2] / sqrt(std::inner_product(infD.begin(), infD.end(), infD.begin(), 0.0)) };
    std::vector<double> AAp = { dot(unitinfD, AC) * unitinfD[0], dot(unitinfD, AC) * unitinfD[1], dot(unitinfD, AC) * unitinfD[2] };

    std::vector<double> Ap = { AAp[0] + A[0], AAp[1] + A[1], AAp[2] + A[2] };

    std::vector<double> boundAAp = _V_AB(A, Ap, C, gamma, ign_col);

    std::vector<double> ApC = { C[0] - Ap[0], C[1] - Ap[1], C[2] - Ap[2] };
    std::vector<double> crss = { infD[1] * ApC[2] - infD[2] * ApC[1],
                                 infD[2] * ApC[0] - infD[0] * ApC[2],
                                 infD[0] * ApC[1] - infD[1] * ApC[0] };
    double mag = sqrt(std::inner_product(crss.begin(), crss.end(), crss.begin(), 0.0)) + (regularize ? core_rad : 0);

    // Checks colinearity
    if (check_collinear(mag, col_crit, ign_col)) {
        if (!ign_col && n_col == 1 && !mute_warning) {
            /*std::cout << "\n\t magsqr:" << magsqr << " \n\t A:(" << A[0] << "," << A[1] << "," << A[2] << ")"
                << " \n\t infD:(" << infD[0] << "," << infD[1] << "," << infD[2] << ")"
                << " \n\t C:(" << C[0] << "," << C[1] << "," << C[2] << ")" << std::endl;*/
        }
        return { 0.0, 0.0, 0.0 };
    }

    double h = mag / sqrt(std::inner_product(infD.begin(), infD.end(), infD.begin(), 0.0));
    std::vector<double> n = { crss[0] / mag, crss[1] / mag, crss[2] / mag };
    std::vector<double> F = { n[0] / h, n[1] / h, n[2] / h };

    if (blobify) {
        F[0] *= gw(h, smoothing_rad);
        F[1] *= gw(h, smoothing_rad);
        F[2] *= gw(h, smoothing_rad);
    }

    if (gamma == 0.0) {
        return { F[0] + boundAAp[0], F[1] + boundAAp[1], F[2] + boundAAp[2] };
    }
    else {
        return { (gamma / (4 * M_PI)) * F[0] + boundAAp[0],
                (gamma / (4 * M_PI)) * F[1] + boundAAp[1],
                (gamma / (4 * M_PI)) * F[2] + boundAAp[2] };
    }
}

std::vector<double> _V_Ainf_in(const std::vector<double>& A, const std::vector<double>& infD,
    const std::vector<double>& C, double gamma, bool ign_col = false) {
    std::vector<double> aux = _V_Ainf_out(A, infD, C, gamma, ign_col);
    return { -aux[0], -aux[1], -aux[2] };
}

Eigen::VectorXd solve(
    const std::vector<VLMSolver::Horseshoe>& HSs,
    const std::vector<Eigen::Vector3d>& Vinfs,
    double t = 0.0,
    std::function<Eigen::Vector3d(const Eigen::Vector3d&, double)> vortexsheet = nullptr,
    std::function<Eigen::Vector3d(int, double)> extraVinf = nullptr
) {
    size_t n = HSs.size();  // Number of horseshoes
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n, n);  // Geometry matrix
    Eigen::VectorXd Vn = Eigen::VectorXd::Zero(n);    // Normal velocity matrix

    // ------------ BUILD MATRICES G AND Vn ------------

    for (size_t i = 0; i < n; ++i) {
        const Horseshoe& hsi = HSs[i];

        // Normal of the panel (cross product and normalization)
        Eigen::Vector3d crss = (Eigen::Map<const Eigen::Vector3d>(hsi.CP.data()) - Eigen::Map<const Eigen::Vector3d>(hsi.A.data()))
            .cross(Eigen::Map<const Eigen::Vector3d>(hsi.B.data()) - Eigen::Map<const Eigen::Vector3d>(hsi.A.data()));
        Eigen::Vector3d nhat = crss.normalized();

        // Iterate over horseshoes
        for (size_t j = 0; j < n; ++j) {
            const Horseshoe& hs = HSs[j];

            // Convert hsi.CP (std::vector<double>) to Eigen::Vector3d
            Eigen::Vector3d CP = Eigen::Map<const Eigen::Vector3d>(hsi.CP.data());

            // Convert Eigen::Vector3d to std::vector<double> before passing to V function
            std::vector<double> CP_vec = { CP[0], CP[1], CP[2] };

            // Call V and get a std::vector<double> result
            std::vector<double> GeomFacVec = V(hs, CP_vec);

            // Convert std::vector<double> to Eigen::Vector3d
            Eigen::Vector3d GeomFac(GeomFacVec[0], GeomFacVec[1], GeomFacVec[2]);

            // Calculate Geometry factors (V function)
            Eigen::Vector3d Gij = (1.0 / (4.0 * M_PI)) * GeomFac;
            double Gijn = Gij.dot(nhat);
            G(i, j) = Gijn;
        }

        // Normal component of freestream velocity
        Eigen::Vector3d this_Vinf = Vinfs[i];
        double Vinfn = this_Vinf.dot(nhat);
        Vn(i) = -Vinfn;

        // Vortex sheet contribution (if applicable)
        if (vortexsheet) {
            Eigen::Vector3d this_Vinfvrtx = vortexsheet(Eigen::Map<const Eigen::Vector3d>(hsi.CP.data()), t);
            Vn(i) += -this_Vinfvrtx.dot(nhat);
        }

        // Extra freestream contribution (if applicable)
        if (extraVinf) {
            Eigen::Vector3d this_extraVinf = extraVinf(i, t);
            Vn(i) += -this_extraVinf.dot(nhat);
        }
    }

    // ------------ SOLVE FOR GAMMA ------------

    Eigen::VectorXd Gamma = G.colPivHouseholderQr().solve(Vn);

    return Gamma;
}

std::vector<double> VLMSolver::V(const Horseshoe& horseshoe, const std::vector<double>& controlPoint,
    bool ignoreCol = false, bool ignoreInfVortex = false, bool onlyInfVortex = false) {
    if (ignoreInfVortex && onlyInfVortex) {
        //std::cerr << "Requested only infinite wake while ignoring infinite wake." << std::endl;
    }

    std::vector<double> Ap = horseshoe.Ap;
    std::vector<double> A = horseshoe.A;
    std::vector<double> B = horseshoe.B;
    std::vector<double> Bp = horseshoe.Bp;
    std::vector<double> CP = horseshoe.CP;
    std::vector<double> infDA = horseshoe.infDA;
    std::vector<double> infDB = horseshoe.infDB;
    double Gamma = horseshoe.Gamma.value_or(0.0);

    std::vector<double> VApA(3, 0.0), VAB(3, 0.0), VBBp(3, 0.0);
    if (!onlyInfVortex) {
        VApA = _V_AB(Ap, A, controlPoint, Gamma, ignoreCol);
        VAB = _V_AB(A, B, controlPoint, Gamma, ignoreCol);
        VBBp = _V_AB(B, Bp, controlPoint, Gamma, ignoreCol);
    }

    std::vector<double> VApinf(3, 0.0), VBpinf(3, 0.0);
    if (!ignoreInfVortex) {
        VApinf = _V_Ainf_in(Ap, infDA, controlPoint, Gamma, ignoreCol);
        VBpinf = _V_Ainf_out(Bp, infDB, controlPoint, Gamma, ignoreCol);
    }

    std::vector<double> V(3, 0.0);
    for (int i = 0; i < 3; ++i) {
        V[i] = VApinf[i] + VApA[i] + VAB[i] + VBBp[i] + VBpinf[i];
    }
    return V;
}
