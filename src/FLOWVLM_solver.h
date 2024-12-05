#pragma once
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>  // Include GLM for glm::vec3
#include <functional>
#include <Eigen/Dense>

// Namespace for VLM Solver
namespace VLMSolver {

    // ------------ STRUCTURE TO REPRESENT HORSESHOES ------------------------------
    struct Horseshoe {
        std::vector<double> Ap, A, B, Bp, CP, infDA, infDB;
        double Gamma;
    };

    std::unordered_map<std::string, std::string> HS_hash = {
    {"Ap", "1"},
    {"A", "2"},
    {"B", "3"},
    {"Bp", "4"},
    {"CP", "5"},
    {"infDA", "6"},
    {"infDB", "7"},
    {"Gamma", "8"}
    };

    // Function declarations
    void _mute_warning(bool booln);
    void _regularize(bool booln);
    void _blobify(bool booln);
    void _smoothing_rad(double val);
    double gw(double r, double sgm);
    bool check_collinear(double magsqr, double col_crit, bool ign_col);
    glm::vec3 vec3FromDoubleVec(const std::vector<double>& vec);

    glm::vec3 V_AB(const std::vector<double>& A, const std::vector<double>& B, const std::vector<double>& C, double Gamma, bool ign_col);
    glm::vec3 V_Ainf_out(const std::vector<double>& A, const std::vector<double>& infD, const std::vector<double>& C, double Gamma, bool ign_col);
    glm::vec3 V_Ainf_in(const std::vector<double>& A, const std::vector<double>& infD, const std::vector<double>& C, double Gamma, bool ign_col);

    // Main solver function
    std::vector<double> solve(
        const std::vector<std::vector<Horseshoe>>& HSs,
        const std::vector<glm::vec3>& Vinfs,
        double t = 0.0,
        std::function<Eigen::Vector3d(const std::vector<double>&, double)> vortexsheet = nullptr,
        std::function<Eigen::Vector3d(size_t, double)> extraVinf = nullptr,
        std::vector<double> extraVinfArgs = {}
    );

    // Helper function for velocity calculation
    std::vector<double> V(Horseshoe& HS, const std::vector<double>& C, bool ign_col = false, bool ign_infvortex = false, bool only_infvortex = false);
}
