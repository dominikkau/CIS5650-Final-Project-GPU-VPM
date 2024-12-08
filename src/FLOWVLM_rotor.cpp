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
/*
#include <vector>
#include <tuple>
#include <cmath>
#include <map>
#include <string>
#include <memory>
#include <iostream>
#include "constants.h"
#include <any>
#include <Eigen/Dense>
#include <glm\detail\type_vec.hpp>
#include <optional>
#include "FLOWVLM_dt.h"

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

    double tol = 0.01; // Tolerance for convergence
    int maxite = 100; // Maximum iterations
    double rlx = 0.0; // Relaxation parameter

    std::vector<std::vector<Eigen::Vector3d>> Vind; // Velocity input
    std::vector<std::vector<double>> Gamma; // Circulation field

    void initialize(int n, double r_lat = 1.0, bool central = false,
        const std::vector<int>& refinement = {}, bool verif = false,
        double figsize_factor = 2.0 / 3.0,
        const std::vector<int>& genblade_args = {},
        const std::vector<int>& rfl_args = {});

    std::vector<double> solvefromCCBlade(
        double Vinf,
        double RPM,
        double rho,
        double t = 0.0,
        bool include_comps = true,
        bool return_performance = false,
        std::optional<double> Vref = std::nullopt,
        std::optional<double> sound_spd = std::nullopt,
        std::optional<std::vector<double>> Uinds = std::nullopt,
        bool _lookuptable = false,
        std::optional<std::vector<std::vector<double>>> _Vinds = std::nullopt,
        int hubtiploss_correction = 0,
        bool AR_to_360extrap = true,
        bool debug = false,
        bool verbosewarn = true);

    std::vector<double> solveFromV(const VindType& Vind, const std::vector<double>& args, bool optargs);

    std::vector<double> solvefromCCBlade(Rotor& self, double Vinf, double RPM, double rho, double t, bool include_comps, bool return_performance, std::optional<double> Vref, std::optional<double> sound_spd, std::optional<std::vector<double>> Uinds, bool _lookuptable, std::optional<std::vector<std::vector<double>>> _Vinds, int hubtiploss_correction, bool AR_to_360extrap, bool debug, bool verbosewarn);

    void setVinf(double Vinf, bool keep_sol = false);
    void setRPM(double RPM);

    std::string save(const std::string& filename, bool addtiproot, bool airfoils, bool wopwop, bool wopbin, const std::string& wopext, double wopv, bool save_horseshoes, const std::vector<std::string>& args);

    void setCoordSystem(const FArrWrap& O, const FMWrap& Oaxis, bool user, Args ...args);

    // Constructor
    Rotor(bool CW, const std::vector<double>& r, const std::vector<double>& chord,
        const std::vector<double>& theta, const std::vector<double>& LE_x,
        const std::vector<double>& LE_z, int B,
        const std::vector<std::tuple<double, std::shared_ptr<Polar>>>& airfoils = {},
        bool turbine_flag = false)
        : 
        CW(CW), 
        r(r), 
        chord(chord), 
        theta(theta), 
        LE_x(LE_x), 
        LE_z(LE_z), 
        B(B),
        airfoils(airfoils), 
        turbine_flag(turbine_flag), 
        RPM(0),
        hubR(r.front()), 
        rotorR(r.back()), 
        m(0), 
        wingsystem(nullptr) {
        // Initialize Polar objects
        _polarroot = std::make_shared<Polar>();  // Replace with a dummy Polar equivalent
        _polartip = std::make_shared<Polar>();
    }

private:
    void _check();
    Blade _generate_blade(int n, double r_lat, bool central, const std::vector<int>& refinement,
        const std::vector<int>& genblade_args,
        std::vector<double>& r, std::vector<double>& chord,
        std::vector<double>& theta,
        std::vector<double>& LE_x, std::vector<double>& LE_z);
    int get_m(const Blade& blade);
    void _verif_discr(const Blade& blade, const std::vector<double>& r,
        const std::vector<double>& chord, const std::vector<double>& theta,
        const std::vector<double>& LE_x, const std::vector<double>& LE_z,
        double figsize_factor);
    void _calc_airfoils(int n, double r_lat, bool central,
        const std::vector<int>& refinement,
        const std::vector<int>& rfl_args);
};

void Rotor::initialize(int n, double r_lat, bool central,
    const std::vector<int>& refinement, bool verif,
    double figsize_factor, const std::vector<int>& genblade_args,
    const std::vector<int>& rfl_args) {
    // Check arguments for consistency
    _check();

    // Check if airfoils are provided
    bool rfl_flag = !airfoils.empty();

    // Generate blade geometry
    std::vector<double> r, chord, theta, LE_x, LE_z;
    Blade blade = _generate_blade(n, r_lat, central, refinement, genblade_args, r, chord, theta, LE_x, LE_z);
    _r = r; _chord = chord; _theta = theta; _LE_x = LE_x; _LE_z = LE_z;
    m = get_m(blade);

    // Verify discretization if required
    if (verif) {
        _verif_discr(blade, r, chord, theta, LE_x, LE_z, figsize_factor);
    }

    // Generate airfoil properties at control points if airfoils are provided
    if (rfl_flag) {
        _calc_airfoils(n, r_lat, central, refinement, rfl_args);
    }

    // Generate the full rotor
    Eigen::Matrix3d blades_Oaxis = CW ?
        (Eigen::Matrix3d() << 0, -1, 0, 0, 0, 1, -1, 0, 0).finished() :
        (Eigen::Matrix3d() << 0, 1, 0, 0, 0, 1, 1, 0, 0).finished();
    double init_angle = 0.0;
    double d_angle = 2 * M_PI / B;

    for (int i = 1; i <= B; ++i) {
        Blade this_blade = (i == 1) ? blade : blade.copy();
        double this_angle = init_angle + (i - 1) * d_angle;

        // Compute the rotation matrix for the current azimuthal angle
        Eigen::Matrix3d rotationMatrix =
            (Eigen::Matrix3d() <<
                cos(this_angle), sin(this_angle), 0,
                -sin(this_angle), cos(this_angle), 0,
                0, 0, 1).finished();

        // Compute the blade's orientation matrix
        Eigen::Matrix3d this_Oaxis = rotationMatrix * blades_Oaxis;

        // Set the blade's coordinate system
        setcoordsystem(this_blade, { 0.0, 0, 0 }, this_Oaxis);

        // Add the blade to the rotor
        addwing(this, "Blade" + std::to_string(i), this_blade, true);
    }

    // Set the rotor in the global coordinate system
    Eigen::Matrix3d rotor_Oaxis;
    rotor_Oaxis << -1, 0, 0, 0, -1, 0, 0, 0, 1;
    setcoordsystem(this->wingsystem, { 0.0, 0, 0 }, rotor_Oaxis);
}

// Constants for hub/tip loss corrections
constexpr std::tuple<double, double, double, double> nohubcorrection = { 1, 0, INFINITY, 5 * std::numeric_limits<double>::epsilon() };
constexpr std::tuple<double, double, double, double> notipcorrection = { 1, 0, INFINITY, 5 * std::numeric_limits<double>::epsilon() };
constexpr std::pair<std::tuple<double, double, double, double>, std::tuple<double, double, double, double>> hubtiploss_nocorrection = { nohubcorrection, notipcorrection };
constexpr std::pair<std::tuple<double, double, double, double>, std::tuple<double, double, double, double>> hubtiploss_correction_prandtl = { {1, 1, 1, 1.0}, {1, 1, 1, 1.0} };
constexpr std::pair<std::tuple<double, double, double, double>, std::tuple<double, double, double, double>> hubtiploss_correction_modprandtl = { {0.6, 5, 0.5, 10}, {2, 1, 0.25, 0.05} };

std::vector<std::vector<Eigen::Vector3d>> calculateInducedVelocity(
    const std::vector<std::vector<Eigen::Vector3d>>& Vind_global,
    bool ignore_infinite_vortex = true);

std::vector<std::vector<double>> solveFromVite() {
    // Previous circulation values
    std::vector<std::vector<double>> prev_sol(B);
    for (int j = 0; j < B; ++j) {
        // Assuming sol["Gamma"] is accessible; replace with your data structure
        prev_sol[j] = Gamma[j];
    }

    int ite = 0;
    double err = 0.0;
    std::vector<std::vector<double>> out;

    for (int i = 1; i <= maxite; ++i) {
        std::vector<std::vector<Eigen::Vector3d>> surfVind;

        if (i == 1) {
            surfVind = Vind; // Initial velocity
        }
        else {
            // Add velocity induced by the lifting surface
            surfVind.resize(B);
            for (int j = 0; j < B; ++j) {
                surfVind[j].resize(Vind[j].size());
                for (size_t k = 0; k < Vind[j].size(); ++k) {
                    surfVind[j][k] = Vind[j][k] +
                        calculateInducedVelocity(Vind, true)[j][k];
                }
            }
        }

        // Solve for the circulation distribution
        out = solveFromV(surfVind);

        // Get the current circulation values
        std::vector<std::vector<double>> this_sol(B);
        for (int j = 0; j < B; ++j) {
            this_sol[j] = Gamma[j]; // Replace with the correct access method
        }

        if (!prev_sol.empty()) {
            // Compute error as the mean variation across blades
            err = 0.0;
            for (int j = 0; j < B; ++j) {
                double bladeError = 0.0;
                for (size_t k = 0; k < prev_sol[j].size(); ++k) {
                    double relativeError = std::abs(prev_sol[j][k] - this_sol[j][k]) /
                        std::abs(prev_sol[j][k]);
                    bladeError += relativeError;
                }
                bladeError /= prev_sol[j].size();
                err += bladeError;
            }
            err /= B;

            if (err < tol) {
                break;
            }

            // Apply relaxation
            for (int j = 0; j < B; ++j) {
                for (size_t k = 0; k < prev_sol[j].size(); ++k) {
                    Gamma[j][k] = rlx * prev_sol[j][k] + (1 - rlx) * this_sol[j][k];
                }
            }
        }

        prev_sol = this_sol;
        ++ite;
    }

    if (ite == maxite) {
        std::cerr << "Warning: Iterative solveFromVite reached max iterations "
            "without converging. "
            << "maxite: " << maxite << "\t error: " << err << "\n";
    }

    return out;
}

using VindType = std::vector<std::vector<glm::vec3>>; // Nested structure for Vind

std::vector<double> Rotor::solveFromV(const VindType& Vind,
    const std::vector<double>& args = {},
    bool optargs = false) {
    // ERROR CASES
    if (Vind.size() != static_cast<size_t>(this->B)) {
        throw std::invalid_argument("Expected " + std::to_string(this->B) +
            " Vind entries; got " + std::to_string(Vind.size()) + ".");
    }

    for (size_t bi = 0; bi < Vind.size(); ++bi) {
        if (Vind[bi].size() != static_cast<size_t>(this->get_mBlade())) {
            throw std::invalid_argument("Expected " + std::to_string(this->get_mBlade()) +
                " Vind[" + std::to_string(bi) + "] entries; got " +
                std::to_string(Vind[bi].size()) + ".");
        }
    }

    // Call solveFromCCBlade with Vind and optional arguments
    return this->solveFromCCBlade(args, Vind, /* lookuptable = true);
}

std::vector<double> Rotor::solvefromCCBlade(
    Rotor& self,
    double Vinf,
    double RPM,
    double rho,
    double t = 0.0,
    bool include_comps = true,
    bool return_performance = false,
    std::optional<double> Vref = std::nullopt,
    std::optional<double> sound_spd = std::nullopt,
    std::optional<std::vector<double>> Uinds = std::nullopt,
    bool _lookuptable = false,
    std::optional<std::vector<std::vector<double>>> _Vinds = std::nullopt,
    int hubtiploss_correction = 0,
    bool AR_to_360extrap = true,
    bool debug = false,
    bool verbosewarn = true) {

    setVinf(Vinf);
    setRPM(RPM);

    // Ensure horseshoe is calculated
    this.getHorseshoe(1);

    if (!sound_spd && verbosewarn) {
        std::cerr << "Warning: No sound speed provided. No Mach corrections will be applied.\n";
    }

    // Calculate distributed loads
    auto [prfrmnc, gammas, mus_drag] = this.calc_distributedloads(
        Vinf, RPM, rho, t, include_comps, return_performance, Vref, sound_spd,
        Uinds, _lookuptable, _Vinds, hubtiploss_correction, AR_to_360extrap, debug);

    // Calculate aerodynamic forces
    auto [gamma, mu_drag] = this.calc_aerodynamicforces(rho, gammas, mus_drag);

    // Initialize fields
    std::vector<std::vector<double>> new_gamma, new_mu, new_Ftot, new_L, new_D, new_S;

    // Format solution fields
    for (int i = 0; i < self.B; ++i) { // Iterate over blades
        for (int j = 0; j < this.get_mBlade(); ++j) { // Iterate over lattices on blade
            new_gamma.push_back(gamma[i][j]);
            new_mu.push_back(mu_drag[i][j]);
            new_Ftot.push_back(self.sol["DistributedLoad"]["field_data"][i][j]);
            new_L.push_back(self.sol["Lift"]["field_data"][i][j]);
            new_D.push_back(self.sol["Drag"]["field_data"][i][j]);
            new_S.push_back(self.sol["RadialForce"]["field_data"][i][j]);
        }
    }

    // Add the fields as FLOWVLM solutions
    this._addsolution(this->wingsystem, "Gamma", new_gamma, t);
    this._addsolution(this->wingsystem, "mu", new_mu, t);
    this._addsolution(this->wingsystem, "Ftot", new_Ftot, t);
    this._addsolution(this->wingsystem, "L", new_L, t);
    this._addsolution(this->wingsystem, "D", new_D, t);
    this._addsolution(this->wingsystem, "S", new_S, t);

    return prfrmnc;
}

void Rotor::setVinf(double Vinf, bool keep_sol = false) {
    // Reset the rotor with the option to keep the solution
    _reset(this, keep_sol);

    // Set Vinf for the rotor's wingsystem with the option to keep the solution
    setVinf(Vinf, keep_sol);
}

void Rotor::setRPM(double RPM) {
    // Reset the rotor's state but keep Vinf
    _reset(true); // Passing true to keep Vinf

    // Perform additional rotor-specific reset
    _resetRotor();

    // Set the RPM value for this rotor
    this->RPM = RPM;
}

std::string Rotor::save(const std::string& filename,
    bool addtiproot,
    bool airfoils,
    bool wopwop,
    bool wopbin,
    const std::string& wopext,
    double wopv,
    bool save_horseshoes,
    const std::vector<std::string>& args) {
    // Ensure wake is calculated correctly if save_horseshoes is enabled
    if (save_horseshoes) {
        getHorseshoe(1);
    }

    // Save the wing system
    std::string result = saveWingSystem(filename, save_horseshoes, args);

    // If airfoils are present, save additional data
    if (!airfoils.empty()) {
        result += saveLoft(filename, addtiproot, airfoils, wopwop, wopbin, wopext, wopv, args);
    }

    return result;
}

void Rotor::setCoordSystem(const FArrWrap& O, const FMWrap& Oaxis, bool user, Args... args) {
    if (user) {
        wingsystem->setCoordSystem(O, Oaxis * glm::mat3{ -1, 0, 0, 0, -1, 0, 0, 0, 1.0 }, args...);
    }
    else {
        wingsystem->setCoordSystem(O, Oaxis, args...);
    }
    resetRotor(true);
}

void Rotor::rotate(const FWrap& degs) {
    glm::mat3 rotOaxis = rotationMatrix(0.0, 0.0, (std::pow(-1, !CW) * degs));
    glm::mat3 newOaxis = rotOaxis * wingsystem->Oaxis;
    setCoordSystem(wingsystem->O, newOaxis);
}

FArrWrap Rotor::getVinfs(FWrap t, const std::string& target,
    std::optional<ExtraVinfFunc> extraVinf, Args... extraVinfArgs) {
    if (VLMSolver::HS_hash.find(target) == VLMSolver::HS_hash.end()) {
        throw std::logic_error("Logic error! Invalid target " + target);
    }

    getHorseshoe(1, t, extraVinf, extraVinfArgs...);

    FArrWrap Vinfs;
    for (int i = 1; i <= B; ++i) {
        auto bladeVinfs = calcInflow(getBlade(i), getRPM(), t, target);
        Vinfs.insert(Vinfs.end(), bladeVinfs.begin(), bladeVinfs.end());
    }

    if (extraVinf) {
        for (int i = 1; i <= B; ++i) {
            auto& blade = getBlade(i);
            for (int j = 1; j <= getM(blade); ++j) {
                Vinfs[i][j] += (*extraVinf)(j, t, extraVinfArgs..., blade);
            }
        }
    }

    return Vinfs;
}

float Rotor::getRPM() const {
    if (!RPM) {
        throw std::runtime_error("RPM not defined yet. Call function `setRPM()` before calling this function.");
    }
    return RPM.value();
}

int Rotor::getMBlade() const {
    return m;
}

Wing& Rotor::getBlade(int bladeIndex) const {
    return getWing(bladeIndex);
}

int Rotor::getM() const {
    return wingsystem->getM();
}

ControlPoint Rotor::getControlPoint(int m) const {
    return wingsystem->getControlPoint(m);
}

Horseshoe Rotor::getHorseshoe(int m, FWrap t, Args... extraVinf) {
    bool flag = std::any_of(
        blades.begin(), blades.end(),
        [](const auto& blade) { return blade->_HSs == nullptr; });

    if (flag) {
        if (!RPM) {
            throw std::runtime_error(
                "RPM hasn't been defined yet. Call function `setRPM()` before calling this function.");
        }
        else if (!wingsystem->Vinf) {
            throw std::runtime_error(
                "Freestream hasn't been defined yet. Call function `setVinf()` before calling this function.");
        }
    }

    if (flag) {
        for (int i = 1; i <= B; ++i) {
            auto& blade = getBlade(i);

            if (!blade->_HSs) {
                blade->O = blade->O; // Center of rotation
                blade->calculateHSs(t, extraVinf...);

                auto VAp = calcInflow(blade, getRPM(), t, "Ap");
                auto VBp = calcInflow(blade, getRPM(), t, "Bp");

                for (size_t j = 0; j < blade->_HSs.size(); ++j) {
                    blade->_HSs[j][6] = VAp[j] / glm::length(VAp[j]);
                    blade->_HSs[j][7] = VBp[j] / glm::length(VBp[j]);
                }
            }
        }
    }

    return wingsystem->getHorseshoe(m, t, extraVinf...);
}

void Rotor::calcInflow(
    const std::function<FArrWrap<>(const FArrWrap<>&, FWrap)>& Vinf,
    FWrap RPM,
    FWrap t /* = 0.0,
    const std::optional<std::vector<FArrWrap<>>>& Vinds /* = std::nullopt) {

    FWrap omega = 2 * M_PI * RPM / 60.0;

    std::vector<std::vector<FArrWrap<>>> data_Vtots; // Inflow in the global c.s.
    std::vector<std::vector<FArrWrap<>>> data_Vccbs; // Inflow in CCBlade's c.s.

    // Iterate over blades
    for (size_t i = 0; i < this->wingsystem.wings.size(); ++i) {
        const auto& blade = this->wingsystem.wings[i];
        std::vector<FArrWrap<>> Vtots;
        std::vector<FArrWrap<>> Vccbs;

        // Iterate over control points
        for (size_t j = 0; j < getM(blade); ++j) {
            auto CP = getControlPoint(blade, j);

            // Freestream velocity in global c.s.
            FArrWrap<> this_Vinf = Vinf(CP, t);

            // Velocity due to rotation in FLOWVLM blade's c.s.
            FArrWrap<> this_Vrot = { omega * this->_r[j], 0.0, 0.0 };
            // Velocity due to rotation in global c.s.
            this_Vrot = counterTransform(this_Vrot, blade.invOaxis, FArrWrap<>(3, 0.0));

            FArrWrap<> this_Vtot = this_Vinf + this_Vrot;

            // Adds any extra induced velocity
            if (Vinds.has_value()) {
                this_Vtot += (*Vinds)[i][j];
            }

            Vtots.push_back(this_Vtot);
            Vccbs.push_back(globalToCCBlade(blade, this_Vtot, this->CW));
        }

        data_Vtots.push_back(Vtots);
        data_Vccbs.push_back(Vccbs);
    }

    // Adds solution fields
    std::unordered_map<std::string, std::variant<std::string, std::vector<std::vector<FArrWrap<>>>>> field_tots = {
        {"field_name", "GlobInflow"},
        {"field_type", "vector"},
        {"field_data", data_Vtots}
    };
    this->sol[std::get<std::string>(field_tots["field_name"])] = field_tots;

    std::unordered_map<std::string, std::variant<std::string, std::vector<std::vector<FArrWrap<>>>>> field_ccbs = {
        {"field_name", "CCBInflow"},
        {"field_type", "vector"},
        {"field_data", data_Vccbs}
    };
    this->sol[std::get<std::string>(field_ccbs["field_name"])] = field_ccbs;
}
*/