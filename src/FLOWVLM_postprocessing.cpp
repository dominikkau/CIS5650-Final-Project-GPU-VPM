//#include <iostream>
//#include <string>
//#include <unordered_map>
//#include <vector>
//#include <stdexcept>
//#include <cmath>
//#include "FLOWVLM_wing.h"
//#include <array>
/*
void calculate_field(Wing& wing, const std::string& field_name,
    std::optional<double> rhoinf = std::nullopt,
    const std::string& qinf = "automatic",
    const std::string& S = "automatic",
    const std::string& l = "automatic",
    const std::string& r_cg = "automatic",
    double t = 0.0, bool lifting_interac = true) {

    // --- ERROR CASES
    // Unknown field
    if (FIELDS.find(field_name) == FIELDS.end()) {
        throw std::invalid_argument("Invalid solution field '" + field_name + "'. Valid fields: { ... }");
    }

    // Missing dependent field
    auto [dependents, field_type] = FIELDS[field_name];
    for (const auto& dep : dependents) {
        if (wing.sol.find(dep) == wing.sol.end()) {
            throw std::runtime_error("Field '" + field_name + "' requested, but '" + dep + "' hasn't been solved yet");
        }
    }

    // AERODYNAMIC FORCES
    if (field_name == "Ftot" || field_name == "L" || field_name == "S" || field_name == "D") {
        if (!rhoinf.has_value()) {
            throw std::runtime_error(field_name + " requested but rhoinf is missing");
        }

        auto [F, SS, D, L] = _calculate_forces(wing, rhoinf.value(), t, lifting_interac);
        wing.addsolution("Ftot", F);
        wing.addsolution("S", SS);
        wing.addsolution("D", D);
        wing.addsolution("L", L);

        // FORCE COEFFICIENTS
    }
    else if (field_name == "CFtot" || field_name == "CL" || field_name == "CS" || field_name == "CD") {
        double _rhoinf;
        double _qinf;
        if (!rhoinf.has_value() && qinf != "automatic") {
            std::cout << field_name << " requested with a given qinf, but rhoinf is missing. Given qinf will be ignored." << std::endl;
            auto Vinf = _aveVinf(wing, t);
            _rhoinf = 1.0;
            _qinf = 0.5 * _rhoinf * dot(Vinf, Vinf);
        }
        else if (qinf == "automatic") {
            auto Vinf = _aveVinf(wing, t);
            _rhoinf = 1.0;
            _qinf = 0.5 * _rhoinf * dot(Vinf, Vinf);
        }
        else {
            _rhoinf = rhoinf.value();
            _qinf = std::stod(qinf);
        }
        double _S = (S == "automatic") ? planform_area(wing) : std::stod(S);

        auto [CF, CS, CD, CL] = _calculate_force_coeffs(wing, _rhoinf, _qinf, _S, t, lifting_interac);
        wing.addsolution("CFtot", CF);
        wing.addsolution("CS", CS);
        wing.addsolution("CD", CD);
        wing.addsolution("CL", CL);

        // AERODYNAMIC FORCE COEFFICIENT PER UNIT SPAN NORMALIZED
    }
    else if (field_name == "Cftot/CFtot" || field_name == "Cd/CD" || field_name == "Cs/CS" || field_name == "Cl/CL") {
        double _rhoinf;
        double _qinf;
        if (!rhoinf.has_value() && qinf != "automatic") {
            std::cout << field_name << " requested with a given qinf, but rhoinf is missing. Given qinf will be ignored." << std::endl;
            auto Vinf = _aveVinf(wing, t);
            _rhoinf = 1.0;
            _qinf = 0.5 * _rhoinf * dot(Vinf, Vinf);
        }
        else if (qinf == "automatic") {
            auto Vinf = _aveVinf(wing, t);
            _rhoinf = 1.0;
            _qinf = 0.5 * _rhoinf * dot(Vinf, Vinf);
        }
        else {
            _rhoinf = rhoinf.value();
            _qinf = std::stod(qinf);
        }
        double _S = (S == "automatic") ? planform_area(wing) : std::stod(S);

        auto [Cf, Cs, Cd, Cl] = _calculate_force_coeffs(wing, _rhoinf, _qinf, _S, true, t, lifting_interac);

        // Converts them into scalars
        std::vector<double> s_Cf, s_Cs, s_Cd, s_Cl;
        std::vector<std::vector<double>> scalars = { s_Cf, s_Cs, s_Cd, s_Cl };
        auto aveVinf = _aveVinf(wing, t);
        for (size_t i = 0; i < scalars.size(); ++i) {
            for (const auto& elem : (i == 0 ? Cf : (i == 1 ? Cs : (i == 2 ? Cd : Cl)))) {
                double sgn = (i == 2) ? sign(dot(elem, aveVinf)) : 1;
                scalars[i].push_back(sgn * norm(elem));
            }
        }

        // Calculates overalls
        auto info = fields_summary(wing);

        // Determines the span of the wing
        auto [_min, _max] = _span_eff(wing);
        double span = _max - _min;

        _addsolution(wing, "Cftot/CFtot", s_Cf / (info["CFtot"] / span));
        _addsolution(wing, "Cs/CS", s_Cs / (info["CS"] / span));
        _addsolution(wing, "Cd/CD", s_Cd / (info["CD"] / span));
        _addsolution(wing, "Cl/CL", s_Cl / (info["CL"] / span));

        // PANEL AREA
    }
    else if (field_name == "A") {
        _addsolution(wing, field_name, _calculate_areas(wing));

        // MOMENTS
    }
    else if (field_name == "Mtot" || field_name == "M_L" || field_name == "M_M" || field_name == "M_N") {
        // Center of gravity
        auto _r_cg = (r_cg == "automatic") ? get_CG(wing) : std::stod(r_cg);

        auto [Mtot, M_L, M_M, M_N] = _calculate_moments(wing, _r_cg);
        _addsolution(wing, "Mtot", Mtot);
        _addsolution(wing, "M_L", M_L);
        _addsolution(wing, "M_M", M_M);
        _addsolution(wing, "M_N", M_N);

        // MOMENTS COEFFICIENTS
    }
    else if (field_name == "CMtot" || field_name == "CM_L" || field_name == "CM_M" || field_name == "CM_N") {
        if (qinf == "automatic" || qinf.empty() || qinf == "0.0") {
            throw std::runtime_error(field_name + " requested but qinf is missing");
        }

        double _l = (l == "automatic") ? get_barc(wing) : std::stod(l);
        double _S = (S == "automatic") ? planform_area(wing) : std::stod(S);

        _addsolution(wing, "CMtot", wing.sol["Mtot"] / (std::stod(qinf) * _S * _l));
        _addsolution(wing, "CM_L", wing.sol["M_L"] / (std::stod(qinf) * _S * _l));
        _addsolution(wing, "CM_M", wing.sol["M_M"] / (std::stod(qinf) * _S * _l));
        _addsolution(wing, "CM_N", wing.sol["M_N"] / (std::stod(qinf) * _S * _l));

        // ERROR CASE
    }
    else {
        throw std::runtime_error("Calculation of " + field_name + " has not been implemented yet!");
    }
}

std::tuple<std::vector<std::array<double, 3>>, std::vector<double>, std::vector<double>, std::vector<double>>
calculateAerodynamicForces(const Wing& wing, const FWrap& rhoinf,
    const FWrap& t = 0.0, bool per_unit_span = false,
    bool lifting_interac = true) {

    std::vector<std::array<double, 3>> totalForces;
    int numberOfHorseshoes = getNumberOfHorseshoes(wing);

    // -------------- CALCULATES TOTAL FORCE totalForces
    // Iterates over horseshoes
    for (int i = 0; i < numberOfHorseshoes; ++i) {
        auto [Ap, A, B, Bp, CP, infDA, infDB, Gamma] = getHorseshoe(wing, i);
        std::array<double, 3> force = { 0.0, 0.0, 0.0 };

        // Iterates over bound vortices of the horseshoe
        std::array<std::array<double, 3>, 3> boundVortices = { {A, B}, {Ap, A}, {B, Bp} };
        for (const auto& BV : boundVortices) {
            // Midway distance
            std::array<double, 3> midpoint = { (BV[0] + BV[1]) / 2.0, (BV[0] + BV[1]) / 2.0, (BV[0] + BV[1]) / 2.0 };
            // Freestream velocity (undisturbed + induced)
            std::array<double, 3> freestreamVelocity = wing.Vinf(midpoint, t);
            if (lifting_interac) {
                freestreamVelocity += inducedVelocity(wing, midpoint, t);
            }
            // Cross product of Vinf and (B-A)
            std::array<double, 3> crossProduct = cross(freestreamVelocity, { BV[1][0] - BV[0][0], BV[1][1] - BV[0][1], BV[1][2] - BV[0][2] });

            double length = per_unit_span ? norm(cross({ freestreamVelocity[0] / norm(freestreamVelocity), freestreamVelocity[1] / norm(freestreamVelocity), freestreamVelocity[2] / norm(freestreamVelocity) }, { BV[1][0] - BV[0][0], BV[1][1] - BV[0][1], BV[1][2] - BV[0][2] })) : 1.0;

            // Force
            for (int j = 0; j < 3; ++j) {
                force[j] += rhoinf * Gamma * crossProduct[j] / length;
            }
        }

        totalForces.push_back(force);
    }

    // Case of calculating drag at the far field
    if (!lifting_interac) {
        auto averageFreestreamVelocity = averageFreestreamVelocity(wing, t);
        auto farFieldForce = calculateForceTrefftz(wing, averageFreestreamVelocity, rhoinf, per_unit_span);
        for (int i = 0; i < numberOfHorseshoes; ++i) {
            for (int j = 0; j < 3; ++j) {
                totalForces[i][j] += farFieldForce[i][j];
            }
        }
    }

    // -------------- DECOMPOSES totalForces INTO COMPONENTS
    auto [S, D, L] = decomposeForces(wing, totalForces, t);

    return { totalForces, S, D, L };
}
*/