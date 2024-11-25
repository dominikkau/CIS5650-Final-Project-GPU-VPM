#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <iostream>

// Structure to hold airfoil data
struct OCCBAirfoilData {
    Eigen::VectorXd cl_spline; // Spline for lift coefficient
    Eigen::VectorXd cd_spline; // Spline for drag coefficient
};

// Function to create a spline from x and y data
Eigen::VectorXd createSpline(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    // Validate input sizes
    if (x.size() < 2 || x.size() != y.size()) {
        throw std::invalid_argument("Invalid input for spline interpolation.");
    }

    // Convert Eigen::VectorXd to std::vector for Boost
    std::vector<double> xVec(x.data(), x.data() + x.size());
    std::vector<double> yVec(y.data(), y.data() + y.size());

    // Create Boost cubic spline
    auto spline = boost::math::interpolators::cardinal_cubic_b_spline<double>(
        yVec.begin(), yVec.end(), xVec[0], xVec[1] - xVec[0]);

    // Generate spline values on the x grid
    Eigen::VectorXd splineResult(x.size());
    for (int i = 0; i < x.size(); ++i) {
        splineResult[i] = spline(xVec[i]);
    }

    return splineResult;
}

// Main function to generate airfoil data
OCCBAirfoilData occb_af_from_data(const std::vector<double>& alpha,
    const std::vector<double>& cl,
    const std::vector<double>& cd,
    int spl_k = 3) {
    // Validate input size
    if (alpha.size() < 2) {
        throw std::invalid_argument("Alpha array too small for spline interpolation.");
    }

    // Convert std::vector to Eigen::VectorXd for compatibility
    Eigen::VectorXd alphaEigen = Eigen::Map<const Eigen::VectorXd>(alpha.data(), alpha.size());
    Eigen::VectorXd clEigen = Eigen::Map<const Eigen::VectorXd>(cl.data(), cl.size());
    Eigen::VectorXd cdEigen = Eigen::Map<const Eigen::VectorXd>(cd.data(), cd.size());

    // Create splines for cl and cd
    Eigen::VectorXd afcl = createSpline(alphaEigen, clEigen);
    Eigen::VectorXd afcd = createSpline(alphaEigen, cdEigen);

    // Return the airfoil data structure
    return { afcl, afcd };
}

//// Example usage
//int main() {
//    try {
//        // Example inputs
//        std::vector<double> alpha = { -10, 0, 10, 20, 30 };
//        std::vector<double> cl = { 0.1, 0.5, 0.8, 0.4, -0.1 };
//        std::vector<double> cd = { 0.02, 0.03, 0.05, 0.07, 0.1 };
//
//        // Create airfoil data
//        OCCBAirfoilData airfoil = occb_af_from_data(alpha, cl, cd);
//
//        // Display results
//        std::cout << "CL spline: " << airfoil.cl_spline.transpose() << std::endl;
//        std::cout << "CD spline: " << airfoil.cd_spline.transpose() << std::endl;
//    }
//    catch (const std::exception& e) {
//        std::cerr << "Error: " << e.what() << std::endl;
//    }
//
//    return 0;
//}
