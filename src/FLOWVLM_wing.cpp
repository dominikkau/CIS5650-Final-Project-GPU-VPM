#include <functional>
#include <numeric>
#include <optional>
#include <iostream>
#include "FLOWVLM_dt.h"
#include "FLOWVLM_solver.h"
#include "FLOWVLM_tools.h"
#include "FLOWVLM.h"

using namespace std;

// Constructor
Wing::Wing(double leftxl_, double lefty_, double leftzl_, double leftchord_, double leftchordtwist_)
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

void Wing::addChord(double x, double y, double z, double c, double twist, int n, double r, bool central,
    const std::vector<std::array<double, 3>>& refinement) {
    // Error cases
    if (c <= 0 || n <= 0 || r <= 0) {
        throw std::invalid_argument("Invalid chord length, number of lattices, or expansion ratio");
    }

    // Reset if existing solution
    if (sol.find("Gamma") != sol.end()) {
        _reset(false, false);
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

void Wing::setcoordsystem(const std::vector<double>& O, const std::vector<std::vector<double>>& Oaxis, bool check) {
    if (check) {
        checkCoordSys(Oaxis);
    }

    this->O = O;
    this->Oaxis = Oaxis;
    this->invOaxis = inverse(Oaxis);
    _reset(false, false);
}

/*void setcoordsystem(const std::vector<double>& O, const std::vector<std::vector<double>>& Oaxis, bool check) {
    int dims = 3;
    std::vector<std::vector<double>> M(dims, std::vector<double>(dims, 0.0));
    for (int i = 0; i < dims; ++i) {
        M[i] = Oaxis[i];
    }
    setcoordsystem(O, M, check);
}*/

void Wing::setVinf(const std::function<std::vector<double>(const std::vector<double>&, double)>& VinfFunc, bool keep_sol) {
    _reset(false, keep_sol);
    this->Vinf = VinfFunc;
}

std::vector<double> Wing::getControlPoint(int m) {
    std::vector<double> CP = { this->xm[m], this->ym[m], this->zm[m] };
    CP = countertransform(CP, this->invOaxis, this->O);
    return CP;
}

std::vector<std::vector<double>> Wing::getVinfs(double t, const std::string& target,
    std::function<std::vector<double>(int, double)> extraVinf,
    const std::vector<double>& extraVinfArgs) {
	VLMSolver s;
    if (s.HS_hash.find(target) == s.HS_hash.end()) {
        throw std::invalid_argument("Logic error! Invalid target " + target);
    }

    int t_i = s.HS_hash[target];

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


Horseshoe Wing::getHorseshoe(int m, double t,
    std::function<std::vector<double>(int, double)> extraVinf) {
        
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

std::vector<double> Wing::getLE(int n) {
    std::vector<double> LE = { this->xlwingdcr[n], this->ywingdcr[n], this->zlwingdcr[n] };
    LE = countertransform(LE, this->invOaxis, this->O);
    return LE;
}

std::vector<double> Wing::getTE(int n) {
    std::vector<double> TE = { this->xtwingdcr[n], this->ywingdcr[n], this->ztwingdcr[n] };
    TE = countertransform(TE, this->invOaxis, this->O);
    return TE;
}

int Wing::get_m() {
    return this->m;
}

Wing Wing::copy() const {
    return Wing(*this);
}

void Wing::_reset(bool keep_Vinf = false, bool keep_sol = false) {
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

void Wing::addsolution(const std::string& field_name, const std::vector<double>& sol_field, double t) {
    this->sol[field_name] = sol_field;
    if (field_name == "Gamma") {
        // _calculateHSs(t); // If needed
        for (size_t i = 0; i < sol_field.size(); ++i) {
            this->HSs[i].Gamma = sol_field[i];
        }
    }
}

std::vector<double> Wing::normalize(const std::vector<double>& v) {
    double magnitude = 0.0;
    for (double component : v) {
        magnitude += component * component;
    }
    magnitude = std::sqrt(magnitude);

    if (magnitude == 0.0) {
        throw std::runtime_error("Cannot normalize a zero-length vector.");
    }

    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] / magnitude;
    }

    return result;
}


void Wing::_calculateHSs(double t = 0.0,
    std::function<std::vector<double>(int, double)> extraVinf) {
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

        infDA = normalize(infDA);
        infDB = normalize(infDB);

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
        Horseshoe hs = { Ap, A, B, Bp, CP, infDA, infDB, Gamma };

        // Add it to the HSs vector
        HSs.push_back(hs);
    }

    this->HSs = HSs;
}


Wing Wing::deepcopy_internal(const Wing& wing) {
    Wing copy = wing;
    return copy;
}


// Method to generate a simple wing
static Wing simpleWing(float b, float ar, float tr, float twist, float lambda, float gamma,
    float twist_tip, int n, float r, bool central,
    std::vector<std::vector<float>> refinement) {

    float cr = 1.0f / tr;    // Root chord
    float c_tip = b / ar;    // Tip chord
    float c_root = cr * c_tip; // Calculating root chord

    // Default twist at the tip
    float twist_t = (twist_tip == -1.0f) ? twist : twist_tip;

    float y_tip = b / 2.0f;
    float x_tip = y_tip * tan(lambda * M_PI / 180.0f);   // X-coordinate of the tip
    float z_tip = y_tip * tan(gamma * M_PI / 180.0f);    // Z-coordinate of the tip

    // Inverting the complex refinement for the opposite wing
    std::vector<std::vector<float>> _ref;
    for (int i = refinement.size() - 1; i >= 0; --i) {
        _ref.push_back({ refinement[i][0], refinement[i][1], 1.0f / refinement[i][2] });
    }

    // Create the wing
    Wing wing = Wing(x_tip, -y_tip, z_tip, c_tip, twist_t);

    // Add root chord
    wing.addChord(0.0f, 0.0f, 0.0f, c_root, twist, n, r, central);

    // Add tip chord
    wing.addChord(x_tip, y_tip, z_tip, c_tip, twist_t, n, r, central);

    return wing;
}

int main(int argc, char* argv[]) {
    // -------- - HORIZONTAL TAIL--------------------
  // Parameters
    double w_b = 40.0,            // Span
        ht_b = w_b * 5 / 16,
        ht_ar = 6.0,
        ht_tr = 0.75,
        ht_twist_root = 0.0,
        ht_twist_tip = -2.5,
        ht_lambda = 15.0,
        ht_gamma = 0.0,
        ht_vt_pos = 0.6,            // Vertical position of horizontal tail on vertical
        ht_c_root = ht_b / ht_ar / ht_tr;
    int n = 1;
    double n_ht = (ceil(n * 20));         // Semi - span of horizontal tail
    double r_ht = 5.0;
    // Creates the lifting surface 
    Wing testWing = simpleWing(ht_b, ht_ar, ht_tr,
        ht_twist_root, ht_lambda, ht_gamma, ht_twist_tip, n_ht, r_ht, false, {});

	FLOWVLM s;
    s.solve(testWing, testWing.Vinf);
    //wing.calculate_field("CFtot", rhoinf)
    //info = fields_summary(wing)
    
}
