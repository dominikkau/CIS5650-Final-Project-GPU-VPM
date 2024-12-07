#include <array>
#include <unordered_map>
#include <stdexcept>
#include <cmath>
#include <string>
#include <functional>
#include <memory>
#include <any>
#include "constants.h"
#include <numeric>
#include "FLOWVLM_dt.h"
#include "FLOWVLM_solver.h"
#include <optional>
#include "FLOWVLM_tools.cpp"
using namespace VLMSolver;

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
    std::vector<Horseshoe> HSs;

    // Constructor
    Wing(double leftxl_, double lefty_, double leftzl_, double leftchord_, double leftchordtwist_)
        : leftxl(leftxl_), lefty(lefty_), leftzl(leftzl_), leftchord(leftchord_), leftchordtwist(leftchordtwist_) {
        m = 0;
        O = { 0.0, 0.0, 0.0 };
        Oaxis = { {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}} };
        invOaxis = { {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}} };
        Vinf = {};
        sol = {};

        xlwingdcr.push_back(leftxl);
        xtwingdcr.push_back(leftxl + leftchord * std::cos(leftchordtwist * M_PI / 180));
        ywingdcr.push_back(lefty);
        zlwingdcr.push_back(leftzl);
        ztwingdcr.push_back(leftzl - leftchord * std::sin(leftchordtwist * M_PI / 180));

        xm = {};
        ym = {};
        zm = {};
        xn.push_back(leftxl + 0.25 * leftchord * std::cos(leftchordtwist * M_PI / 180));
        yn.push_back(lefty);
        zn.push_back(leftzl - 0.25 * leftchord * std::sin(leftchordtwist * M_PI / 180));
        HSs = {};
    }

    void addChord(double x, double y, double z, double c, double twist, int n, double r = 1.0, bool central = false,
        const std::vector<std::array<double, 3>>& refinement = {}) {
        // Error cases
        if (c <= 0 || n <= 0 || r <= 0) {
            throw std::invalid_argument("Invalid chord length, number of lattices, or expansion ratio");
        }

        // Reset if existing solution
        if (sol.find("Gamma") != sol.end()) {
            _reset();
        }

        double twistRadians = twist * M_PI / 180;

        // Refinement pre-calculations
        std::vector<int> ns; // Number of lattices in each section
        double ntot = 0.0, ctot = 0.0;

        if (!refinement.empty()) {
            int nsecs = refinement.size();
            ntot = std::accumulate(refinement.begin(), refinement.end(), 0.0,
                [](double sum, const auto& section) {
                auto [chord, weight, ratio] = section; // Unpack array
                return sum + weight;
            });

            ctot = std::accumulate(refinement.begin(), refinement.end(), 0.0,
                [](double sum, const auto& section) {
                auto [chord, weight, ratio] = section; // Unpack array
                return sum + chord;
            });

            ns.clear();
            ns.reserve(nsecs);

            for (int i = 0; i < nsecs; ++i) {
                if (i == nsecs - 1 && nsecs != 1) {
                    ns.push_back(n - std::accumulate(ns.begin(), ns.end(), 0));
                }
                else {
                    auto& [chord, weight, ratio] = refinement[i];
                    ns.push_back(static_cast<int>(std::floor(n * weight / ntot)));
                }
            }
        }

        // Boundary points
        std::vector<double> Ll = { xlwingdcr.back(), ywingdcr.back(), zlwingdcr.back() };
        std::vector<double> Lt = { xtwingdcr.back(), ywingdcr.back(), ztwingdcr.back() };
        std::vector<double> Rl = { x, y, z };
        std::vector<double> Rt = { x + c * std::cos(twistRadians), y, z - c * std::sin(twistRadians) };

        double length = std::sqrt(std::inner_product(Rl.begin(), Rl.end(), Ll.begin(), 0.0, std::plus<>(),
            [](double a, double b) { return (a - b) * (a - b); }));

        double cumulativeLength = 0.0;

        for (int i = 0; i < n; ++i) {
            double len;

            if (!refinement.empty()) {
                int sec_i = 0;
                for (int j = 0; j < ns.size(); ++j) {
                    if (i >= std::accumulate(ns.begin(), ns.begin() + j + 1, 0)) {
                        sec_i++;
                    }
                    else {
                        break;
                    }
                }

                int prev_ns = (sec_i == 0) ? 0 : std::accumulate(ns.begin(), ns.begin() + sec_i, 0);
                int current_n = ns[sec_i];
                auto& [chord, weight, ratio] = refinement[sec_i];
                double current_r = ratio;             // Use the ratio as current_r
                double section_length = length * (chord / ctot);  // Use the chord as section_length

                int local_i = i - prev_ns;

                double p = section_length / ((current_n * (current_n - 1) / 2) * (current_r + 1) / (current_r - 1));
                double d1 = p * (current_n - 1) / (current_r - 1);
                len = d1 + p * local_i;
            }
            else if (r == 1.0) {
                len = length / n;
            }
            else {
                if (!central) {
                    double p = length / ((n * (n - 1) / 2) * (r + 1) / (r - 1));
                    double d1 = p * (n - 1) / (r - 1);
                    len = d1 + p * i;
                }
                else {
                    double centralRatio = (central == true) ? 0.5 : central;
                    if (i < std::floor(n * centralRatio)) {
                        double leftLength = length * centralRatio;
                        int leftCount = std::floor(n * centralRatio);
                        double leftRatio = r;

                        double p = leftLength / ((leftCount * (leftCount - 1) / 2) * (leftRatio + 1) / (leftRatio - 1));
                        double d1 = p * (leftCount - 1) / (leftRatio - 1);
                        len = d1 + p * i;
                    }
                    else {
                        double rightLength = length * (1 - centralRatio);
                        int rightCount = n - std::floor(n * centralRatio);
                        double rightRatio = 1 / r;

                        int local_i = i - std::floor(n * centralRatio);
                        double p = rightLength / ((rightCount * (rightCount - 1) / 2) * (rightRatio + 1) / (rightRatio - 1));
                        double d1 = p * (rightCount - 1) / (rightRatio - 1);
                        len = d1 + p * local_i;
                    }
                }
            }

            cumulativeLength += len;

            std::vector<double> THISl = {
                Ll[0] + (cumulativeLength / length) * (Rl[0] - Ll[0]),
                Ll[1] + (cumulativeLength / length) * (Rl[1] - Ll[1]),
                Ll[2] + (cumulativeLength / length) * (Rl[2] - Ll[2])
            };
            std::vector<double> THISt = {
                Lt[0] + (cumulativeLength / length) * (Rt[0] - Lt[0]),
                Lt[1] + (cumulativeLength / length) * (Rt[1] - Lt[1]),
                Lt[2] + (cumulativeLength / length) * (Rt[2] - Lt[2])
            };

            xlwingdcr.push_back(THISl[0]);
            ywingdcr.push_back(THISl[1]);
            zlwingdcr.push_back(THISl[2]);
            xtwingdcr.push_back(THISt[0]);
            ztwingdcr.push_back(THISt[2]);

            std::vector<double> Cn = { THISt[0] - THISl[0], THISt[1] - THISl[1], THISt[2] - THISl[2] };
            std::vector<double> N = { THISl[0] + 0.25 * Cn[0], THISl[1] + 0.25 * Cn[1], THISl[2] + 0.25 * Cn[2] };
            xn.push_back(N[0]);
            yn.push_back(N[1]);
            zn.push_back(N[2]);

            std::vector<double> Cm = {
                Lt[0] + ((cumulativeLength - len / 2) / length) * (Rt[0] - Lt[0]) - (Ll[0] + ((cumulativeLength - len / 2) / length) * (Rl[0] - Ll[0])),
                Lt[1] + ((cumulativeLength - len / 2) / length) * (Rt[1] - Lt[1]) - (Ll[1] + ((cumulativeLength - len / 2) / length) * (Rl[1] - Ll[1])),
                Lt[2] + ((cumulativeLength - len / 2) / length) * (Rt[2] - Lt[2]) - (Ll[2] + ((cumulativeLength - len / 2) / length) * (Rl[2] - Ll[2]))
            };
            std::vector<double> M = { Cm[0] * 0.75, Cm[1] * 0.75, Cm[2] * 0.75 };
            xm.push_back(M[0]);
            ym.push_back(M[1]);
            zm.push_back(M[2]);
        }

        if (std::abs(cumulativeLength - length) / length > 1e-10) {
            throw std::logic_error("Critical logic error! cumulativeLength != length");
        }

        m += n;
    }

    void setcoordsystem(const std::vector<double>& O, const std::vector<std::vector<double>>& Oaxis, bool check = true) {
        if (check) {
            checkCoordSys(Oaxis);
        }

        this->O = O;
        this->Oaxis = Oaxis;
        this->invOaxis = inverse(Oaxis);
        _reset();
    }

    void setcoordsystem(const std::vector<double>& O, const std::vector<std::vector<double>>& Oaxis, bool check = true) {
        int dims = 3;
        std::vector<std::vector<double>> M(dims, std::vector<double>(dims, 0.0));
        for (int i = 0; i < dims; ++i) {
            M[i] = Oaxis[i];
        }
        setcoordsystem(O, M, check);
    }

    void setVinf(const std::function<std::vector<double>(const std::vector<double>&, double)>& VinfFunc, bool keep_sol = false) {
        _reset(keep_sol);
        this->Vinf = VinfFunc;
    }

    std::vector<double> getControlPoint(int m) {
        std::vector<double> CP = { this->xm[m], this->ym[m], this->zm[m] };
        CP = countertransform(CP, this->invOaxis, this->O);
        return CP;
    }

    std::vector<std::vector<double>> getVinfs(double t = 0.0, const std::string& target = "CP",
        std::function<std::vector<double>(int, double)> extraVinf = nullptr,
        const std::vector<double>& extraVinfArgs = {}) {

        if (VLMSolver::HS_hash.find(target) == VLMSolver::HS_hash.end()) {
            throw std::invalid_argument("Logic error! Invalid target " + target);
        }

        int t_i = VLMSolver::HS_hash[target];

        // Calculates Vinf at each control point
        std::vector<std::vector<double>> Vinfs;

        for (int i = 0; i < get_m(); ++i) {
            std::vector<double> T;

            if (target == "CP") {
                T = getControlPoint(i);  // Get control point data
            }
            else {
                // Get the horseshoe object
                Horseshoe hs = getHorseshoe(i, t);

                // Dynamically access the correct vector from the horseshoe struct
                std::vector<double> selectedVector;
                switch (t_i) {
                case 1: selectedVector = hs.Ap; break;
                case 2: selectedVector = hs.A; break;
                case 3: selectedVector = hs.B; break;
                case 4: selectedVector = hs.Bp; break;
                case 5: selectedVector = hs.CP; break;
                case 6: selectedVector = hs.infDA; break;
                case 7: selectedVector = hs.infDB; break;
                case 8:  // If Gamma is selected, handle accordingly
                    if (hs.Gamma) {
                        selectedVector.push_back(hs.Gamma.value());  // Optional<double> value
                    }
                    break;
                default:
                    throw std::invalid_argument("Invalid target index for Horseshoe.");
                }

                T = selectedVector;  // Assign the selected vector to T
            }

            // Compute Vinf using the selected T
            std::vector<double> this_Vinf = Vinf(T, t);
            if (extraVinf) {
                this_Vinf = extraVinf(i, t);
            }
            Vinfs.push_back(this_Vinf);
        }

        return Vinfs;
    }


    Horseshoe getHorseshoe(int m, double t = 0.0,
        std::function<std::vector<double>(int, double)> extraVinf = nullptr) {
        
        // ERROR CASES
        if (m <= 0 || m > this->m) {
            throw std::out_of_range("Invalid m (m>self.m or m<=0)");
        }
        else if (this->sol.find("Gamma") == this->sol.end() && this->Vinf == nullptr) {
            throw std::runtime_error("Freestream hasn't been defined yet, please call function setVinf()");
        }

        if (this->HSs.empty()) {
            _calculateHSs(t, extraVinf);
        }

        return this->HSs[m];
    }

    std::vector<double> getLE(int n) {
        std::vector<double> LE = { this->xlwingdcr[n], this->ywingdcr[n], this->zlwingdcr[n] };
        LE = countertransform(LE, this->invOaxis, this->O);
        return LE;
    }

    std::vector<double> getTE(int n) {
        std::vector<double> TE = { this->xtwingdcr[n], this->ywingdcr[n], this->ztwingdcr[n] };
        TE = countertransform(TE, this->invOaxis, this->O);
        return TE;
    }

    int get_m() {
        return this->m;
    }

    Wing copy() const {
        return Wing(*this);
    }


private:
    void _reset(bool keep_Vinf = false, bool keep_sol = false) {
        if (!keep_sol) {
            this->sol.clear();
        }
        else {
            // Keep the sol entries except "Gamma"
            for (auto it = this->sol.begin(); it != this->sol.end(); ) {
                if (it->first != "Gamma") {
                    ++it;
                }
                else {
                    it = this->sol.erase(it);
                }
            }
        }

        if (!keep_Vinf) {
            this->Vinf = nullptr;  // Reset the function
        }


        this->HSs.clear();
    }

    void addsolution(const std::string& field_name, const std::vector<double>& sol_field, double t = 0.0) {
        this->sol[field_name] = sol_field;
        if (field_name == "Gamma") {
            // _calculateHSs(t); // If needed
            for (size_t i = 0; i < sol_field.size(); ++i) {
                this->HSs[i].Gamma = sol_field[i];
            }
        }
    }

    void _calculateHSs(double t = 0.0, 
        std::function<std::vector<double>(int, double)> extraVinf = nullptr) {
        std::vector<Horseshoe> HSs;

        for (int i = 0; i < get_m(); ++i) {
            // Horseshoe geometry
            std::vector<double> Ap = { this->xtwingdcr[i], this->ywingdcr[i], this->ztwingdcr[i] };
            std::vector<double> A = { this->xn[i], this->yn[i], this->zn[i] };
            std::vector<double> B = { this->xn[i + 1], this->yn[i + 1], this->zn[i + 1] };
            std::vector<double> Bp = { this->xtwingdcr[i + 1], this->ywingdcr[i + 1], this->ztwingdcr[i + 1] };

            // Transform points to global coordinates
            Ap = countertransform(Ap, this->invOaxis, this->O);
            A = countertransform(A, this->invOaxis, this->O);
            B = countertransform(B, this->invOaxis, this->O);
            Bp = countertransform(Bp, this->invOaxis, this->O);

            // Control point
            std::vector<double> CP = getControlPoint(i);

            // Direction of semi-infinite vortices
            std::vector<double> infDA = this->Vinf(Ap, t);
            std::vector<double> infDB = this->Vinf(Bp, t);

            // Extra freestream
            if (extraVinf) {
                infDA = extraVinf(i, t);
                infDB = extraVinf(i, t);
            }

            infDA = glm::normalize(infDA);
            infDB = glm::normalize(infDB);

            // Circulation
            std::optional<double> Gamma = std::nullopt;

            // Check if "Gamma" exists in the map
            if (this->sol.count("Gamma")) {
                try {
                    // Attempt to cast the value to a double
                    Gamma = std::any_cast<double>(this->sol.at("Gamma"));
                }
                catch (const std::bad_any_cast& e) {
                    // Handle the case where the type does not match
                    std::cerr << "Error: Unable to cast 'Gamma' to double: " << e.what() << std::endl;
                }
            }


            // Create the Horseshoe object
            Horseshoe hs = { Ap, A, B, Bp, CP, infDA, infDB, Gamma};

            // Add it to the HSs vector
            HSs.push_back(hs);
        }

        this->HSs = HSs;
    }

    Wing deepcopy_internal(const Wing& wing) {
        Wing copy = wing;
        return copy;
    }


};
