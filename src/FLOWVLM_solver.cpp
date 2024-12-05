#include <iostream>
#include <vector>
#include <cmath>
#include <glm/glm.hpp>  // Include GLM for glm::vec3
#include "constants.h"
#include "FLOWVLM_solver.h"

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

void _mute_warning(bool booln) {
    mute_warning = booln;
}

void _regularize(bool booln) {
    regularize = booln;
}

void _blobify(bool booln) {
    blobify = booln;
}

void _smoothing_rad(double val) {
    smoothing_rad = val;
}

// Wincklman's regularizing function
double gw(double r, double sgm) {
    double ratio = r / sgm;
    return std::pow(ratio, 3) * (std::pow(ratio, 2) + 2.5) / std::pow((std::pow(ratio, 2) + 1), 2.5);
}

// ------------ HELPER FUNCTIONS -----------------------------------------------
bool check_collinear(double magsqr, double col_crit, bool ign_col) {
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

// Helper function to convert std::vector<double> to glm::vec3
glm::vec3 vec3FromDoubleVec(const std::vector<double>& vec) {
    if (vec.size() >= 3) {
        return glm::vec3(static_cast<float>(vec[0]), static_cast<float>(vec[1]), static_cast<float>(vec[2]));
    }
    return glm::vec3(0.0f);  // Default in case the vector is too small
}

// ------------ VORTEX FUNCTIONS ----------------------------------------------
glm::vec3 V_AB(const std::vector<double>& A, const std::vector<double>& B, const std::vector<double>& C, double Gamma, bool ign_col) {
    glm::vec3 r0 = vec3FromDoubleVec(B) - vec3FromDoubleVec(A);
    glm::vec3 r1 = vec3FromDoubleVec(C) - vec3FromDoubleVec(A);
    glm::vec3 r2 = vec3FromDoubleVec(C) - vec3FromDoubleVec(B);

    glm::vec3 crss = glm::cross(r1, r2);  // Use glm::cross
    double magsqr = glm::dot(crss, crss) + (regularize ? smoothing_rad : 0);

    if (check_collinear(magsqr / glm::length(r0), col_crit, ign_col)) {
        return glm::vec3(0.0f);
    }

    glm::vec3 F1 = crss / static_cast<float>(magsqr);  // Divide by magsqr (cast to float)
    glm::vec3 aux = glm::normalize(r1) - glm::normalize(r2);
    double F2 = glm::dot(r0, aux);

    if (blobify) {
        F1 *= gw(glm::length(crss) / glm::length(r0), smoothing_rad);
    }

    if (Gamma == 0) {
        return F1 * glm::vec3(F2);
    }
    else {
        return glm::vec3(F1.x * F2, F1.y * F2, F1.z * F2) * glm::vec3((Gamma / (4 * M_PI)));
    }
}

glm::vec3 V_Ainf_out(const vector<double>& A, const vector<double>& infD, const vector<double>& C, double Gamma, bool ign_col) {
    glm::vec3 AC(C[0] - A[0], C[1] - A[1], C[2] - A[2]);
    glm::vec3 unitinfD = glm::normalize(glm::vec3(infD[0], infD[1], infD[2]));
    glm::vec3 AAp = unitinfD * glm::dot(unitinfD, AC);
    AAp += glm::vec3(A[0], A[1], A[2]);

    std::vector<double> AAp_vec = { AAp.x, AAp.y, AAp.z };
    glm::vec3 boundAAp = V_AB(A, AAp_vec, C, Gamma, ign_col);
    glm::vec3 ApC = glm::vec3(C[0] - AAp[0], C[1] - AAp[1], C[2] - AAp[2]);
    glm::vec3 crss = glm::cross(vec3FromDoubleVec(infD), ApC);
    double mag = glm::length(crss) + (regularize ? smoothing_rad : 0);

    if (check_collinear(mag, col_crit, ign_col)) {
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }

    double h = mag / glm::length(vec3FromDoubleVec(infD));
    glm::vec3 n = glm::normalize(crss) / static_cast<float>(h);
    glm::vec3 F = n;

    if (blobify) {
        F *= gw(h, smoothing_rad);
    }

    if (Gamma == 0) {
        return F + boundAAp;
    }
    else {
        return F * glm::vec3(Gamma / (4 * M_PI)) + boundAAp;
    }
}

glm::vec3 V_Ainf_in(const vector<double>& A, const vector<double>& infD, const vector<double>& C, double Gamma, bool ign_col) {
    return V_Ainf_out(A, infD, C, Gamma, ign_col) * -1.0f;
}

vector<double> V(VLMSolver::Horseshoe& HS, const vector<double>& C, bool ign_col = false, bool ign_infvortex = false, bool only_infvortex = false) {
    vector<double> result(3, 0.0);

    // Decompose HS
    vector<double>& Ap = HS.Ap;
    vector<double>& A = HS.A;
    vector<double>& B = HS.B;
    vector<double>& Bp = HS.Bp;
    vector<double>& CP = HS.CP;
    vector<double>& infDA = HS.infDA;
    vector<double>& infDB = HS.infDB;
    double Gamma = HS.Gamma;

    glm::vec3 VApA, VAB, VBBp, VApinf, VBpinf;

    if (!only_infvortex) {
        VApA = V_AB(Ap, A, C, Gamma, ign_col);
        VAB = V_AB(A, B, C, Gamma, ign_col);
        VBBp = V_AB(B, Bp, C, Gamma, ign_col);
    }

    if (!ign_infvortex) {
        VApinf = V_Ainf_in(Ap, infDA, C, Gamma, ign_col);
        VBpinf = V_Ainf_out(Bp, infDB, C, Gamma, ign_col);
    }

    // Accumulate induced velocities
    result[0] = VApA.x + VAB.x + VBBp.x + VApinf.x + VBpinf.x;
    result[1] = VApA.y + VAB.y + VBBp.y + VApinf.y + VBpinf.y;
    result[2] = VApA.z + VAB.z + VBBp.z + VApinf.z + VBpinf.z;

    return result;
}
