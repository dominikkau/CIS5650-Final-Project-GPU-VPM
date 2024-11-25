#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <optional>
#include "constants.h"


// Define OCCBAirfoilData with appropriate types for splines
struct OCCBAirfoilData {
    boost::math::interpolators::cardinal_cubic_b_spline<double> cl_spline;
    boost::math::interpolators::cardinal_cubic_b_spline<double> cd_spline;

    // Constructor
    OCCBAirfoilData(
        const boost::math::interpolators::cardinal_cubic_b_spline<double>& cl,
        const boost::math::interpolators::cardinal_cubic_b_spline<double>& cd)
        : cl_spline(cl), cd_spline(cd) {}
};

// Define the Rotor geometry structure
struct OCCBRotor {
    std::vector<double> r;         // Radial locations (m)
    std::vector<double> chord;     // Chord lengths (m)
    std::vector<double> theta;     // Total twist including pitch (rad)
    std::vector<OCCBAirfoilData> af; // Airfoil data for each section
    double Rhub;                   // Hub radius (m)
    double Rtip;                   // Tip radius (m)
    int B;                         // Number of blades
    double precone;                // Precone angle (rad)

    // Constructor for easy initialization
    OCCBRotor(const std::vector<double>& r_,
        const std::vector<double>& chord_,
        const std::vector<double>& theta_,
        const std::vector<OCCBAirfoilData>& af_,
        double Rhub_, double Rtip_, int B_, double precone_)
        : r(r_), chord(chord_), theta(theta_), af(af_), Rhub(Rhub_), Rtip(Rtip_), B(B_), precone(precone_) {}
};

// Define the operating point structure for turbine/propeller
struct OCCBInflow {
    std::vector<double> Vx;  // Axial inflow velocity (m/s)
    std::vector<double> Vy;  // Tangential inflow velocity (m/s)
    double rho;              // Air density (kg/m^3)

    // Constructor for easy initialization
    OCCBInflow(const std::vector<double>& Vx_,
        const std::vector<double>& Vy_,
        double rho_)
        : Vx(Vx_), Vy(Vy_), rho(rho_) {}
};

// Function to create an airfoil spline from alpha, cl, and cd arrays
OCCBAirfoilData occb_af_from_data(const std::vector<double>& alpha,
    const std::vector<double>& cl,
    const std::vector<double>& cd,
    int spl_k = 3) {
    // Ensure alpha has enough points for spline interpolation
    if (alpha.size() < 2) {
        throw std::invalid_argument("Alpha array too small for spline interpolation.");
    }

    int k = std::min(static_cast<int>(alpha.size()) - 1, spl_k); // Determine spline degree

    // Convert alpha to radians
    std::vector<double> alpha_rad(alpha.size());
    for (size_t i = 0; i < alpha.size(); ++i) {
        alpha_rad[i] = alpha[i] * M_PI / 180.0;
    }

    // Spline creation with smoothing approximation
    // (Boost's spline doesn't directly support smoothing; use approximate smoothing or preprocess data)
    auto afcl = boost::math::interpolators::cardinal_cubic_b_spline<double>(
        cl.data(), cl.size(), alpha_rad.front(), (alpha_rad.back() - alpha_rad.front()) / (alpha.size() - 1));

    auto afcd = boost::math::interpolators::cardinal_cubic_b_spline<double>(
        cd.data(), cd.size(), alpha_rad.front(), (alpha_rad.back() - alpha_rad.front()) / (alpha.size() - 1));

    return OCCBAirfoilData{ afcl, afcd };
}

// Function to evaluate airfoil spline at a given alpha
std::pair<double, double> occb_airfoil(const OCCBAirfoilData& af, double alpha) {
    // Convert alpha to radians
    double alpha_rad = alpha * M_PI / 180.0;

    // Evaluate splines
    double cl = af.cl_spline(alpha_rad);
    double cd = af.cd_spline(alpha_rad);

    return { cl, cd };
}


// Define the FLOWVLM2OCCBlade function
OCCBRotor FLOWVLM2OCCBlade(
    const FLOWVLM& self, // Assuming FLOWVLM is a class representing the FLOWVLM data structure
    double RPM,
    size_t blade_i,
    bool turbine_flag,
    std::optional<double> sound_spd = std::nullopt,
    bool AR_to_360extrap = true,
    double CDmax = 1.3,
    std::vector<Polar>* out_polars = nullptr,
    std::vector<OCCBAirfoilData>* out_ccb_polars = nullptr)
{
    // Error cases
    if (self.airfoils.size() < 2) {
        throw std::runtime_error("Airfoil data not found when generating CCBlade Rotor.");
    }
    else if (self._polars.empty()) {
        throw std::runtime_error("Control point polars haven't been calculated yet. "
            "Run `_calc_airfoils()` before calling this function.");
    }
    else if (self.sol.find("CCBInflow") == self.sol.end()) {
        throw std::runtime_error("CCBInflow field not found. Call `calc_inflow()` before calling this function.");
    }

    // Initialize variables
    double Rhub = self.hubR;
    double Rtip = self.rotorR;
    double precone = 0.0;
    auto inflows = self.sol.at("CCBInflow").field_data.at(blade_i);

    std::vector<OCCBAirfoilData> af;

    // Process each polar
    for (size_t i = 0; i < self._polars.size(); ++i) {
        auto& polar = self._polars[i];
        double r_over_R = self._r[i] / Rtip;
        double c_over_r = self._chord[i] / self._r[i];

        // Compute inflow parameters
        double this_Vinf = std::abs(inflows[i][0]);
        std::optional<double> tsr = (this_Vinf < 1e-4)
            ? std::nullopt
            : std::optional<double>((2 * M_PI * RPM / 60.0 * Rtip) / this_Vinf);

        // Mach correction
        Polar this_polar = polar;
        if (sound_spd) {
            double Ma = std::sqrt(
                std::pow(inflows[i][0], 2) +
                std::pow(inflows[i][1], 2) +
                std::pow(inflows[i][2], 2)) / *sound_spd;
            if (Ma >= 1) {
                throw std::runtime_error("Mach correction requested on Ma >= 1.0");
            }
            auto [alpha, cl] = ap.get_cl(polar);
            this_polar = ap.Polar(
                ap.get_Re(polar),
                alpha,
                cl / std::sqrt(1 - std::pow(Ma, 2)),
                ap.get_cd(polar)[1],
                ap.get_cm(polar)[1]);
        }

        // Apply 3D corrections
        this_polar = ap.correction3D(this_polar, r_over_R, c_over_r, tsr);

        // 360-degree extrapolation
        if (AR_to_360extrap) {
            auto c_spline1D = Spline1D(self._r / self.rotorR, self._chord, 1);
            double c_75 = c_spline1D(0.75);
            double AR = c_75 / self.rotorR;
            this_polar = ap.extrapolate(this_polar, CDmax, AR);
        }
        else {
            this_polar = ap.extrapolate(this_polar, CDmax);
        }

        // Ensure polar is injective
        this_polar = ap.injective(this_polar);

        // Save to output polars if specified
        if (out_polars) {
            out_polars->push_back(this_polar);
        }

        // Convert to OCCBAirfoilData
        auto [alpha, cl] = ap.get_cl(this_polar);
        auto [_, cd] = ap.get_cd(this_polar);
        OCCBAirfoilData ccb_polar = occb_af_from_data(alpha, cl, cd, 5);

        if (out_ccb_polars) {
            out_ccb_polars->push_back(ccb_polar);
        }

        af.push_back(ccb_polar);
    }

    // Create the OCCBRotor object
    Eigen::VectorXd theta = (-1.0) * ((turbine_flag ? -1.0 : 1.0) * self.CW * self._theta * M_PI / 180.0);
    return OCCBRotor(self._r, self._chord, theta, af, Rhub, Rtip, self.B, precone);
}







int main() {
    try {
        // Example inputs
        std::vector<double> alpha = { -10, 0, 10, 20, 30 };
        std::vector<double> cl = { 0.1, 0.5, 0.8, 0.4, -0.1 };
        std::vector<double> cd = { 0.02, 0.03, 0.05, 0.07, 0.1 };

        // Create airfoil data
        OCCBAirfoilData airfoil = occb_af_from_data(alpha, cl, cd);

        // Evaluate airfoil properties at alpha = 15 degrees
        auto [cl_val, cd_val] = occb_airfoil(airfoil, 15.0);

        std::cout << "At alpha = 15 degrees:" << std::endl;
        std::cout << "CL = " << cl_val << ", CD = " << cd_val << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}

