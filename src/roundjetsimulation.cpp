#include <iostream>
#include <cmath>
#include <vector>
#include <utility>
#include <glm/glm.hpp>
#include "roundjetsimulation.hpp"
#include "vpmcore/vpmmain.h"

vpm::pidx_t addAnnulus(ParticleBuffer& particleBuffer, vpm::real circulation, vpm::real R,
    int Nphi, vpm::real sigma, vpm::real area, vpm::vec3 jetOrigin,
    vpm::mat3 jetOrientation, bool isStatic, vpm::pidx_t startingIndex, vpm::pidx_t maxParticles) {
        
        // Arclength corresponding to phi for circle with radius r
        auto fun_S = [](vpm::real phi, vpm::real r) { return r * phi; };
        // Circle circumference
        vpm::real Stot = fun_S(2 * PI, R);

        // Non-dimensional arc length from 0 to a given value <=1
        auto fun_s = [fun_S, Stot](vpm::real phi, vpm::real r) { return fun_S(phi, r) / Stot; };

        // Angle associated to a given non-dimensional arc length
        auto fun_phi = [](vpm::real s) { return 2 * PI * s; };

        // Length of a given filament in a cross-sectional cell
        auto fun_length = [fun_S, R](vpm::real r, vpm::real tht, vpm::real phi1, vpm::real phi2) {
            vpm::real S1 = fun_S(phi1, R + r * cos(tht));
            vpm::real S2 = fun_S(phi2, R + r * cos(tht));
            return S2 - S1;
        };
        
        auto fun_X_global = [jetOrigin, jetOrientation](vpm::vec3 x) {
        return jetOrigin + jetOrientation * x;
        };

        auto fun_Gamma_global = [jetOrientation](vpm::vec3 Gamma) {
        return jetOrientation * Gamma;
        };

        // Perimeter spacing between cross sections
        vpm::real dS = Stot / Nphi;
        
        // Non-dimensional perimeter spacing
        vpm::real ds = dS / Stot;

        vpm::pidx_t idx = startingIndex;
        // Discretization of annulus into cross-sections
        for (int i = 0; i < Nphi; i++){
            
            // Non-dimensional arc-length position of cross section along centerline
            vpm::real sc1 = ds * i;        // Lower bound
            vpm::real sc2 = ds *(i+1);     // Upper bound
            vpm::real sc = (sc1 + sc2)/2;  // Center

            // Angle of cross section along centerline
            vpm::real phi1 = fun_phi(sc1);       // Lower bound
            vpm::real phi2 = fun_phi(sc2);       // Upper bound
            vpm::real phic = fun_phi(sc);        // Center

            vpm::vec3 Xc{ R * sin(phic), R * cos(phic), 0 };      // Center of the cross section
            vpm::vec3 T{ -cos(phic), sin(phic), 0 };              // Unitary tangent of the cross section
            vpm::mat3 Naxis;
            Naxis[0] = T;
            Naxis[1] = glm::cross(vpm::vec3(0, 0, 1), T);
            Naxis[2] = vpm::vec3(0, 0, 1);

            // Position
            vpm::vec3 X = Xc;

            // Filament length
            vpm::real length = fun_length(0, R, phi1, phi2);

            // Vortex strength
            vpm::vec3 Gamma = circulation * length * T;

            if (idx >= maxParticles - 1) return idx;

            particleBuffer.X(idx) = fun_X_global(X);
            const vpm::vec3 Gamma_global = fun_Gamma_global(Gamma);
            particleBuffer.GammaX(idx) = Gamma_global.x;
            particleBuffer.GammaY(idx) = Gamma_global.y;
            particleBuffer.GammaZ(idx) = Gamma_global.z;
            //particleBuffer.circulation(idx) = circulation;
            particleBuffer.sigma(idx) = sigma;
            //particleBuffer.vol(idx) = area * length;
            particleBuffer.index(idx) = idx;
            //particleBuffer.isStatic()[idx] = isStatic;
            ++idx;
        }
        return idx;
}


std::pair<vpm::pidx_t, vpm::pidx_t> initRoundJet(ParticleBuffer& particleBuffer, ParticleBuffer& boundaryBuffer,
    vpm::pidx_t maxParticles) {

    // ------- SIMULATION PARAMETERS ------- 
    // (m) jet diameter
    const vpm::real d{ 45.4e-3f };
    vpm::real U1 = 40.0f;
    // (m/s) Coflow velocity
    vpm::real U2 = 0.0f; 
    // (deg) Coflow angle from centerline
    vpm::vec3 U2angle= vpm::vec3 {0.0f};
    //  Ratio of inflow momentum thickness of shear layer to diameter, θ/d
    vpm::real thetaod = 0.05;
    // Maximum sigmas in z-direction to create annulis for defining BC
    vpm::real max_zsigma = 2.0f;
    // Threshold at which not to add particles
    vpm::real minWfraction = 0.02f;
    // Origin of jet
    vpm::vec3 jetOrigin = vpm::vec3 {0.0f};
    // orientation of jet
    vpm::mat3 jetOrientation = vpm::mat3 {1.0f};        // Identity matrix

    // -------  SOLVER OPTIONS ------- 
    int steps_per_d = 100;           // Number of time steps for the centerline at U1 to travel one diameter
    int d_travel_tot = 60;          // Run simulation for an equivalent of this many diameters
    vpm::real maxRoR = 1.0f;            // (m) maximum radial distance to discretize
    vpm::real dxotheta = 0.5f;        // Distance Δx between particles over momentum thickness θ
    vpm::real overlap = 2.4f;           // Overlap between particles

    vpm::pidx_t numParticles{ 0 };

    // Define freestream (coflow) velocity
    vpm::vec3 Vfreestream = vpm::vec3{ 0, 0, U2 };

    // TODO: How to initialize Uinf = Vinf in pfield

    vpm::real R = d/2;                         // (m) jet radius
    vpm::vec3 Cline = jetOrientation[2];        // Centerline direction

    // Temporal discretization
    vpm::real dt = d / steps_per_d / U1;         // (s) time step
    int nsteps = static_cast<int>(std::ceil(d_travel_tot * d / U1 / dt));       // Number of time steps

    // Spatial discretization
    vpm::real maxR       = maxRoR * R;
    vpm::real dx         = dxotheta * thetaod * d;     // (m) approximate distance between particles
    vpm::real sigma      = overlap * dx;               // particle smoothing

    // Top-hat velocity profile with smooth edges
    auto Vprofile = [d](vpm::real r, vpm::real theta) {
        return std::abs(r) < d / 2 ? std::tanh((d / 2 - std::abs(r)) / theta) : 0.0;
    };
    
    // Vjet lambda function
    auto Vjet = [U1, d, thetaod, Vprofile](vpm::real r) {
        return U1 * Vprofile(r, thetaod * d);
    };
    
    // -------  SIMULATION SETUP ------- 
    // auto Vjet_wrap = [Vjet](vpm::vec3 X){ Vjet(X[1])};
    
    // Convert velocity profile to vorticity profile
    auto dVdr = [d, thetaod, U1](vpm::real r) {
        return U1 * r / (pow(cosh((d - 2 * abs(r))/(2 * thetaod * d)), 2) * thetaod * d * abs(r));
    };

    auto Wr = [dVdr](vpm::real r){ return -dVdr(r);};
    
    // Brute-force find maximum vorticity in the region to discretize
    int length = 1000;
    vpm::real step = (2 * maxR)/(length - 1);
    vpm::real Wpeak = -FLT_MAX;

    for (vpm::real radius = -maxR; radius <= maxR; radius += step) {
        Wpeak = fmax(Wr(radius), Wpeak);
    }
    
    // Number of cross sections
    int Nphi = static_cast<int>(std::ceil(2 * PI * R / dx));
    // Number of radial sections (annuli)
    int NR = static_cast<int>(std::ceil(maxR / dx));
    // (m) actual radial distance between particles
    vpm::real dr = maxR / NR;

    // Axial component of the freestream
    vpm::real V2 = glm::dot(Vfreestream, Cline);

    // Boundary condition indices (needed only if we're removing the others before running simul)
    std::vector<int> BCi;
    //Particle * boundaryParticles;

    int startingIndex { 0 };
    // Spatial discretization of the boundary condition
    for (int ri = 1; ri <= NR; ++ri) {      // Iterate over annuli
        
        // Annulus lower, upper bounds and center
        vpm::real rlo = dr * (ri - 1);
        vpm::real rup = dr * ri;
        vpm::real rc = (rlo + rup) / 2;

        // Velocity at center of annulus
        vpm::real Vc = V2 + Vjet(rc);
        // Distance traveled in one time step
        vpm::real dz = Vc * dt;

        // Integrate vorticity radially over annulus segment
        // TODO: Confirm: implement closed form solution for -vJet
        vpm::real Wint = -(Vjet(rup) - Vjet(rlo));

        // Annulus circulation
        vpm::real circulation = Wint * dz + 1e-12f;
        // Mean vorticity
        vpm::real Wmean = Wint / (rup - rlo);

        // Area of annulus swept
        vpm::real area = dz * (rup - rlo);

        // Number of longitudinal divisions
        int Nz = static_cast<int>(std::ceil(max_zsigma * sigma / dz));

        if (abs(Wmean) / Wpeak >= minWfraction) {
            // Iterate over Z layers (time steps)
            for (int zi = 0; zi <= Nz; ++zi) {

                int org_np = numParticles;

                vpm::vec3 currentJetOrigin = jetOrigin + zi * dz * Cline;
               
                bool isStatic = zi!=0;

                // Call addAnnulus with appropriate arguments
                startingIndex = addAnnulus(particleBuffer, circulation, R, Nphi, sigma, area,
                    currentJetOrigin, jetOrientation, isStatic, startingIndex, maxParticles);
                
                if (startingIndex == -1) break;
  
                numParticles = startingIndex;

                // If zi == 0, update boundary condition indices
                if (zi == 0) {
                    for (int pi = org_np; pi < numParticles; ++pi) {
                        BCi.push_back(pi);
                    }
                }

            }
        }
    }   
    vpm::pidx_t j = 0;
    // BCi always the same in
    for (vpm::pidx_t i = 0; i < BCi.size(); i++){
        boundaryBuffer.X(j) = particleBuffer.X(BCi[i]);
        boundaryBuffer.GammaX(j) = particleBuffer.GammaX(BCi[i]);
        boundaryBuffer.GammaY(j) = particleBuffer.GammaY(BCi[i]);
        boundaryBuffer.GammaZ(j) = particleBuffer.GammaZ(BCi[i]);
        boundaryBuffer.sigma(j) = particleBuffer.sigma(BCi[i]);
        boundaryBuffer.index(j) = particleBuffer.index(BCi[i]);
        j++;
    }
    // remove all particles from particleBuffer that are not in the BCi array?
    // need to return initial boundary particle buffer
    return {numParticles, static_cast<vpm::pidx_t>(BCi.size())}; // or BCi.size();
}