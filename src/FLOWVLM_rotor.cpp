//################################################################################
//# ROTOR CLASS
//################################################################################
/*
`Rotor(CW, r, chord, theta, LE_x, LE_z, B, airfoil)`

Object defining the geometry of a rotor / propeller / wind turbine.This class
behaves as an extension of the WingSystem class, hence all functions of
WingSystem can be applied to a Rotor object.

# Arguments
* CW::Bool                   : True for clockwise rotation, false for CCW.
* r::Array{ Float64,1 } : Radius position for the following variables.
* chord::Array{ Float64,1 } : Chord length.
* theta::Array{ Float64,1 } : Angle of attack(deg) from the rotor's plane
of rotation.
* LE_x::Array{ Float64,1 } : x - position of leading edge(positive is ahead
    of radial axis relative to rotation).
    * LE_z::Array{ Float64,1 } : z - position of leading edge(height from plane
        of rotation).
    * B::Int64 : Number of blades.

    # Optional Arguments
    * airfoils::Array{ Tuple{Float64, airfoilprep.Polar},1 } : 2D airfoil properties
    along blade in the form[(r_i, Polar_i)]
    with Polar_i describes the airfoil at i - th
    radial position r_i(both the airfoil geometry
        in Polar_i and r_i must be normalized).At
    least root(r = 0) and tip(r = 1) must be given
    so all positions in between can be
    extrapolated.This properties are only used
    when calling CCBlade and for generating good
    loking visuals; ignore if only solving the VLM.

    NOTE: r here is the radial position after precone is included in the geometry,
    hence the need of explicitely declaring LE_z.

    # PROPERTIES
    * `sol` : Contains solution fields specific for Rotor types.They are formated
    as sol[field_name] = Dict(
        "field_name" = > output_field_name,
        "field_type" = > "scalar" or "vector",
        "field_data" = > data
    )
    where `data` is an array data[i] = [val1, val2, ...] containing
this field values(scalar or vector) of all control points in the
i - th blade.

< !--NOTE TO SELF : r is the y - direction on a wing, hence, remember to build the
    blade from root in the direction of positive y. -->
    */

#include <vector>
#include <tuple>
#include <cmath>
#include <map>
#include <string>
#include <memory>
#include <iostream>
#include "constants.h"

    // Forward declarations for external dependencies.
class Polar;  // Placeholder for `ap.Polar`
class WingSystem;  // Placeholder for `WingSystem`

// Rotor class definition
class Rotor {
public:
    // Constructor arguments
    bool CW;  // Clockwise rotation
    std::vector<double> r;  // Radius positions
    std::vector<double> chord;  // Chord length
    std::vector<double> theta;  // Angle of attack (degrees)
    std::vector<double> LE_x;  // Leading edge x-coordinates
    std::vector<double> LE_z;  // Leading edge z-coordinates
    int B;  // Number of blades
    std::vector<std::tuple<double, std::shared_ptr<Polar>>> airfoils;  // Airfoil properties
    bool turbine_flag;

    // Properties
    double RPM;  // Revolutions per minute
    double hubR;  // Hub radius
    double rotorR;  // Rotor radius
    int m;  // Number of control points per blade
    std::map<std::string, std::any> sol;  // Solution fields (e.g., CCBlade outputs)

    // Internal data storage
    WingSystem* wingsystem;  // Pointer to a WingSystem instance
    std::vector<double> _r, _chord, _theta, _LE_x, _LE_z;
    std::vector<std::shared_ptr<Polar>> _polars;
    std::shared_ptr<Polar> _polarroot, _polartip;

    // Constructor
    Rotor(bool CW, const std::vector<double>& r, const std::vector<double>& chord,
        const std::vector<double>& theta, const std::vector<double>& LE_x,
        const std::vector<double>& LE_z, int B,
        const std::vector<std::tuple<double, std::shared_ptr<Polar>>>& airfoils = {},
        bool turbine_flag = false)
        : CW(CW), r(r), chord(chord), theta(theta), LE_x(LE_x), LE_z(LE_z), B(B),
        airfoils(airfoils), turbine_flag(turbine_flag), RPM(0),
        hubR(r.front()), rotorR(r.back()), m(0), wingsystem(nullptr) {
        // Initialize Polar objects
        _polarroot = std::make_shared<Polar>();  // Replace with a dummy Polar equivalent
        _polartip = std::make_shared<Polar>();
    }

    // Function to initialize the geometry
    void initialize(int n, double r_lat = 1.0, bool central = false, bool verif = false) {
        // Generate blade geometry
        generateBlade(n, r_lat, central);

        // Configure and add blades to the rotor
        double d_angle = 2 * M_PI / B;
        for (int i = 0; i < B; ++i) {
            double this_angle = i * d_angle;
            configureBlade(i, this_angle);
        }

        // Set the global coordinate system for the rotor
        setGlobalCoordinateSystem();
    }

private:
    void generateBlade(int n, double r_lat, bool central) {
        // Placeholder for generating blade geometry and storing it in internal variables
        // (e.g., _r, _chord, _theta, _LE_x, _LE_z).
    }

    void configureBlade(int bladeIndex, double angle) {
        // Placeholder for configuring blade orientation and position in the rotor system
    }

    void setGlobalCoordinateSystem() {
        // Placeholder for setting the rotor in the global coordinate system
    }
};

// Constants for hub/tip loss corrections
constexpr std::tuple<double, double, double, double> nohubcorrection = { 1, 0, INFINITY, 5 * std::numeric_limits<double>::epsilon() };
constexpr std::tuple<double, double, double, double> notipcorrection = { 1, 0, INFINITY, 5 * std::numeric_limits<double>::epsilon() };
constexpr std::pair<std::tuple<double, double, double, double>, std::tuple<double, double, double, double>> hubtiploss_nocorrection = { nohubcorrection, notipcorrection };
constexpr std::pair<std::tuple<double, double, double, double>, std::tuple<double, double, double, double>> hubtiploss_correction_prandtl = { {1, 1, 1, 1.0}, {1, 1, 1, 1.0} };
constexpr std::pair<std::tuple<double, double, double, double>, std::tuple<double, double, double, double>> hubtiploss_correction_modprandtl = { {0.6, 5, 0.5, 10}, {2, 1, 0.25, 0.05} };
