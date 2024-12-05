#pragma once
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>  // Include GLM for glm::vec3

// Namespace for VLM Solver
namespace VLMSolver {

    // ------------ STRUCTURE TO REPRESENT HORSESHOES ------------------------------
    //# HORSESHOE
    //# A horseshoe is defined as a 5 - segments vortex by the array
    //# HS = [Ap, A, B, Bp, CP, infDA, infDB, Gamma], with
    //#
    //#   * `Ap::Array{Float64,1}`    : A-side trailing edge.
    //#   * `A::Array{Float64,1}`     : A-side of the bound vortex.
    //#   * `B::Array{Float64,1}`     : B-side of the bound vortex.
    //#   * `Bp::Array{Float64,1}`    : B-side trailing edge.
    //#   * `CP::Array{Float64,1}`    : Control point of the lattice associated to the HS.
    //#   * `infDA::Array{Float64,1}` : Direction of A-side semi-infinite vortex.
    //#   * `infDB::Array{Float64,1}` : Direction of B-side semi-infinite vortex.
    //#   * `Gamma::Float64 or nothing`: Circulation of the horseshoe.
    //#
    //# infDA and infDB must be unitary vectors pointing from the trailing edge in
    //# the direction of infinite.
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

    struct Horseshoe {
        std::vector<double> Ap, A, B, Bp, CP, infDA, infDB;
        double Gamma;
    };

    glm::vec3 V_Ainf_out(const vector<double>& A, const vector<double>& infD, const vector<double>& C, double Gamma, bool ign_col);
    glm::vec3 V_Ainf_in(const vector<double>& A, const vector<double>& infD, const vector<double>& C, double Gamma, bool ign_col);
    vector<double> V(VLMSolver::Horseshoe& HS, const vector<double>& C, bool ign_col = false, bool ign_infvortex = false, bool only_infvortex = false);
}