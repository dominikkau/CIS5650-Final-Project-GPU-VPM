#ifndef WING_H
#define WING_H

#include <array>
#include <unordered_map>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <string>
#include <memory>
#include <any>
#include "constants.h"
#include "FLOWVLM_dt.h"
#include "FLOWVLM_solver.h"
#include <optional>
#include "FLOWVLM_tools.cpp"

    using namespace std;

    class Wing {
    public:
        // Constructor Parameters
        double leftxl;                                  // x-position of leading edge of the left tip
        double lefty;                                   // y-position of left tip
        double leftzl;                                  // z-position of leading edge of the left tip
        double leftchord;                               // Chord of the left tip
        double leftchordtwist;                          // Twist of the left tip's chord in degrees

        // Properties
        int m;                                          // Number of lattices
        std::vector<double> O;                          // Origin of local reference frame
        std::vector<std::vector<double>> Oaxis;         // Unit vectors of local reference frame
        std::vector<std::vector<double>> invOaxis;      // Inverse unit vectors
        std::function<std::vector<double>(const std::vector<double>&, double)> Vinf; // Vinf

        // Data storage
        std::unordered_map<std::string, std::any> sol;  // Dictionary storing solved fields
        std::vector<double> xlwingdcr;                  // x-position of leading edge
        std::vector<double> xtwingdcr;                  // x-position of trailing edge
        std::vector<double> ywingdcr;                   // y-position of the chord
        std::vector<double> zlwingdcr;                  // z-position of leading edge
        std::vector<double> ztwingdcr;                  // z-position of trailing edge

        std::vector<double> xm;                         // x-position of control point
        std::vector<double> ym;                         // y-position of control point
        std::vector<double> zm;                         // z-position of control point
        std::vector<double> xn;                         // x-position of bound vortex
        std::vector<double> yn;                         // y-position of bound vortex
        std::vector<double> zn;                         // z-position of bound vortex

        // Calculation data
        std::vector<VLMSolver::Horseshoe> HSs;

        // Constructor
        Wing(double leftxl_, double lefty_, double leftzl_, double leftchord_, double leftchordtwist_);

        // Methods
        void addChord(double x, double y, double z, double c, double twist, int n, double r = 1.0, bool central = false,
            const std::vector<std::array<double, 3>>& refinement = {});

        void setcoordsystem(const std::vector<double>& O, const std::vector<std::vector<double>>& Oaxis, bool check = true);
        void setVinf(const std::function<std::vector<double>(const std::vector<double>&, double)>& VinfFunc, bool keep_sol = false);

        std::vector<double> getControlPoint(int m);
        std::vector<std::vector<double>> getVinfs(double t = 0.0, const std::string& target = "CP",
            std::function<std::vector<double>(int, double)> extraVinf = nullptr,
            const std::vector<double>& extraVinfArgs = {});

        VLMSolver::Horseshoe getHorseshoe(int m, double t = 0.0,
            std::function<std::vector<double>(int, double)> extraVinf = nullptr);

        std::vector<double> getLE(int n);
        std::vector<double> getTE(int n);

        int get_m();
        void addsolution(const std::string& field_name, const std::vector<double>& sol_field, double t = 0.0);

    private:
        void _reset(bool keep_sol = false);
        void _calculateHSs(double t, std::function<std::vector<double>(int, double)> extraVinf = nullptr);
        void checkCoordSys(const std::vector<std::vector<double>>& Oaxis);
        std::vector<double> countertransform(const std::vector<double>& vec, const std::vector<std::vector<double>>& matrix, const std::vector<double>& origin);
        std::vector<std::vector<double>> inverse(const std::vector<std::vector<double>>& matrix);
    };

#endif // WING_H
