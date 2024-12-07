#pragma once
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>  // Include GLM for glm::vec3
#include <functional>
#include <Eigen/Dense>
#include <optional>

// Namespace for VLM Solver
namespace VLMSolver {

    // ------------ STRUCTURE TO REPRESENT HORSESHOES ------------------------------
    struct Horseshoe {
        std::vector<double> Ap, A, B, Bp, CP, infDA, infDB;
        std::optional<double> Gamma;
    };

    extern std::unordered_map<std::string, int> HS_hash;

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
        const std::vector<VLMSolver::Horseshoe>& HSs,
        std::vector<std::vector<double>>& Vinfs,
        double t = 0.0,
        std::function<Eigen::Vector3d(const std::vector<double>&, double)> vortexsheet = nullptr,
        std::function<Eigen::Vector3d(size_t, double)> extraVinf = nullptr,
        std::vector<double> extraVinfArgs = {}
    );

    // Helper function for velocity calculation
    std::vector<double> V(const Horseshoe& horseshoe, const std::vector<double>& controlPoint,
        bool ignoreCol = false, bool ignoreInfVortex = false, bool onlyInfVortex = false);
}
