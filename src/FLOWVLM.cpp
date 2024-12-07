#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
#include <functional>

#include "FLOWVLM_dt.h"
#include "FLOWVLM_solver.h"
#include "FLOWVLM_wing.h"
#include "FLOWVLM_wingsystem.h"
#include "FLOWVLM_postprocessing.h"
#include "FLOWVLM_rotor_ccb.h"
#include "FLOWVLM_rotor.h"
#include "FLOWVLM_tools.h"

// Define a more manageable type alias for field data
using FieldType = std::pair<std::vector<std::string>, std::string>;

static const std::unordered_map<std::string, FieldType> FIELDS = {
        {"Gamma", { {}, "scalar" }},         // Vortex strength
        {"Vinf", { {}, "vector" }},         // Velocity at each CP used for Gamma
        // ################## LIFT AND SIDEWASH ####################
        {"Ftot", { {"Gamma"}, "vector" }},     // Aerodynamic force (D+L+S)
        {"D", { {"Gamma"}, "vector" }},     // Drag
        {"L", { {"Gamma"}, "vector" }},     // Lift
        {"S", { {"Gamma"}, "vector" }},     // Sideslip force
        {"CFtot", { {"Gamma"}, "vector" }},     // COEFFICIENTS PER PANEL
        {"CD", { {"Gamma"}, "vector" }},     //
        {"CL", { {"Gamma"}, "vector" }},     //
        {"CS", { {"Gamma"}, "vector" }},     //
        {"Cftot/CFtot", { {"CFtot"}, "scalar" }}, // NORMALIZED UNIT-SPAN
        {"Cd/CD", { {"CFtot"}, "scalar" }},     // COEFFICIENTS PER PANEL
        {"Cl/CL", { {"CFtot"}, "scalar" }},     //
        {"Cs/CS", { {"CFtot"}, "scalar" }},     //
        // ################# INDUCED DRAG ###########################
        // {"Dind", { {"Gamma"}, "vector" }},  // Induced drag
        // {"CDind", { {"Gamma"}, "vector" }},  // Induced drag coefficient
        // ################## MOMENTS ###############################
        {"A", { {}, "scalar" }},         // Area of each panel
        {"Mtot", { {"Ftot"}, "vector" }},  // Total moment (-M_L + M_M - M_N)
        {"M_L", { {"Ftot"}, "vector" }},  // Rolling moment
        {"M_M", { {"Ftot"}, "vector" }},  // Pitching moment
        {"M_N", { {"Ftot"}, "vector" }},  // Yawing moment
        {"CMtot", { {"Mtot"}, "vector" }},  // COEFFICIENTS PER PANEL
        {"CM_L", { {"Mtot"}, "vector" }},  //
        {"CM_M", { {"Mtot"}, "vector" }},  //
        {"CM_N", { {"Mtot"}, "vector" }},  //
        // ################## EXTRA FIELDS ####################
        {"mu", { {}, "scalar" }},        // Dipole strength of dragging line
        {"Vind", { {}, "vector" }},        // Any induced velocity field
        {"Vvpm", { {}, "vector" }},        // Velocity induced by VPM
        {"Vvpm_ApA", { {}, "vector" }},        // Velocity induced by VPM
        {"Vvpm_AB", { {}, "vector" }},        // Velocity induced by VPM
        {"Vvpm_BBp", { {}, "vector" }},        // Velocity induced by VPM
        {"Vkin", { {}, "vector" }},        // Kinematic velocity
        {"ftot", { {}, "vector" }},        // Aerodynamic force (D+L+S) per unit span
        {"default-vector", { {}, "vector" }},  // Place holder for a vector field
        {"default-scalar", { {}, "scalar" }}   // Place holder for a scalar field
};

class FLOWVLM {
public:
    // ------------ CONSTANTS -----------------------------------
    const double pm = 3.0 / 4.0; // Default chord-position of the control point
    const double pn = 1.0 / 4.0; // Default chord-position of the bound vortex

    // WING AND WINGSYSTEM COMMON FUNCTIONS

    // Solves the VLM of the Wing or WingSystem
    void solve(Wing& wing,
        std::function<std::vector<double>(const std::vector<double>&, double)> Vinf,
        double t = 0.0,
        std::function<Eigen::Vector3d(const std::vector<double>&, double)> vortexsheet = nullptr,
        std::function<std::vector<double>(int, double)> extraVinf = nullptr,
        bool keep_sol = false,
        const std::vector<double>& extraVinfArgs = {}) {

        // Sets Vinf (this forces to recalculate horseshoes)
        wing.setVinf(Vinf, keep_sol);

        // Obtain horseshoes
        std::vector<VLMSolver::Horseshoe> HSs = getHorseshoes(wing, t, extraVinf, extraVinfArgs);
        std::vector<std::vector<double>> Vinfs = wing.getVinfs(t, "", extraVinf, extraVinfArgs);

        // Flatten Vinfs (std::vector<std::vector<double>>) into a single std::vector<double>
        std::vector<double> flattenedVinfs;
        for (const auto& vec : Vinfs) {
            flattenedVinfs.insert(flattenedVinfs.end(), vec.begin(), vec.end());
        }

        // Calls the solver
        auto Gammas = VLMSolver::solve(HSs, Vinfs, t, vortexsheet);

        // Add solutions to wing
        wing.addsolution("Gamma", Gammas, t);
        wing.addsolution("Vinf", flattenedVinfs, t); // Use flattenedVinfs
    }

    // Returns all the horseshoes of the Wing or WingSystem
    std::vector<VLMSolver::Horseshoe> getHorseshoes(Wing& wing, double t = 0.0,
        std::function<std::vector<double>(int, double)> extraVinf = {},
        const std::vector<double>& extraVinfArgs = {}) {

        int m = wing.get_m();
        std::vector<VLMSolver::Horseshoe> HSs;
        for (int i = 1; i <= m; ++i) {
            HSs.push_back(wing.getHorseshoe(i, t, extraVinf));
        }
        return HSs;
    }

    // Returns the velocity induced at point X
    std::vector<double> Vind(Wing& wing, const std::vector<double>& X, double t = 0.0,
        bool ign_col = false, bool ign_infvortex = false,
        bool only_infvortex = false) {

        std::vector<double> V(3, 0.0);
        // Adds the velocity induced by each horseshoe
        for (int i = 1; i <= wing.get_m(); ++i) {
            auto HS = wing.getHorseshoe(i, t);
            std::vector<double> inducedVelocity = VLMSolver::V(HS, X, ign_col, ign_infvortex, only_infvortex);
            for (size_t i = 0; i < V.size(); ++i) {
                V[i] += inducedVelocity[i];
            }
        }
        return V;
    }

    std::string get_hash(const std::string& var) {
        auto it = VLMSolver::HS_hash.find(var);
        if (it != VLMSolver::HS_hash.end()) {
            return std::to_string(it->second);
        }
        else {
            return "Key not found";
        }
    }

};
