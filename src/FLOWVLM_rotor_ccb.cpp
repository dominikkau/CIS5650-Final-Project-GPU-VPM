#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


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

