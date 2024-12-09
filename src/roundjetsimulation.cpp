#include <iostream>
#include <cmath>
#include <vector>
#include <utility>
#include <glm/glm.hpp>
#include "roundjetsimulation.hpp"
#include "vpmcore/kernel.h"
#include <iostream>

int addAnnulus(Particle* particleBuffer, vpmfloat circulation, vpmfloat R,
    int Nphi, vpmfloat sigma, vpmfloat area, vpmvec3 jetOrigin,
    vpmmat3 jetOrientation, bool isStatic, int startingIndex, int maxParticles){
        
        // Arclength corresponding to phi for circle with radius r
        auto fun_S = [](vpmfloat phi, vpmfloat r) { return r * phi; };
        // Circle circumference
        vpmfloat Stot = fun_S(2 * PI, R);

        // Non-dimensional arc length from 0 to a given value <=1
        auto fun_s = [fun_S, Stot](vpmfloat phi, vpmfloat r) { return fun_S(phi, r) / Stot; };

        // Angle associated to a given non-dimensional arc length
        auto fun_phi = [](vpmfloat s) { return 2 * PI * s; };

        // Length of a given filament in a cross-sectional cell
        auto fun_length = [fun_S, R](vpmfloat r, vpmfloat tht, vpmfloat phi1, vpmfloat phi2) {
            vpmfloat S1 = fun_S(phi1, R + r * cos(tht));
            vpmfloat S2 = fun_S(phi2, R + r * cos(tht));
            return S2 - S1;
        };
        
        auto fun_X_global = [jetOrigin, jetOrientation](vpmvec3 x) {
        return jetOrigin + jetOrientation * x;
        };

        auto fun_Gamma_global = [jetOrientation](vpmvec3 Gamma) {
        return jetOrientation * Gamma;
        };

        // Perimeter spacing between cross sections
        vpmfloat dS = Stot / Nphi;
        
        // Non-dimensional perimeter spacing
        vpmfloat ds = dS / Stot;

        int idx = startingIndex;
        // Discretization of annulus into cross-sections
        for (int i = 0; i < Nphi; i++){
            
            // Non-dimensional arc-length position of cross section along centerline
            vpmfloat sc1 = ds * i;        // Lower bound
            vpmfloat sc2 = ds *(i+1);     // Upper bound
            vpmfloat sc = (sc1 + sc2)/2;  // Center

            // Angle of cross section along centerline
            vpmfloat phi1 = fun_phi(sc1);       // Lower bound
            vpmfloat phi2 = fun_phi(sc2);       // Upper bound
            vpmfloat phic = fun_phi(sc);        // Center

            vpmvec3 Xc{ R * sin(phic), R * cos(phic), 0 };      // Center of the cross section
            vpmvec3 T{ -cos(phic), sin(phic), 0 };              // Unitary tangent of the cross section
            vpmmat3 Naxis;
            Naxis[0] = T;
            Naxis[1] = glm::cross(vpmvec3(0, 0, 1), T);
            Naxis[2] = vpmvec3(0, 0, 1);

            // Position
            vpmvec3 X = Xc;

            // Filament length
            vpmfloat length = fun_length(0, R, phi1, phi2);

            // Vortex strength
            vpmvec3 Gamma = circulation * length * T;

            if (idx >= maxParticles - 1) return -1;

            particleBuffer[idx].X = fun_X_global(X);
            particleBuffer[idx].Gamma = fun_Gamma_global(Gamma);
            particleBuffer[idx].circulation = circulation;
            particleBuffer[idx].sigma = sigma;
            particleBuffer[idx].vol = area * length;
            particleBuffer[idx].index = idx;
            particleBuffer[idx].isStatic = isStatic;
            ++idx;
        }
        return idx;
}


std::pair<int, int> initRoundJet(Particle* particleBuffer, Particle* boundaryBuffer, int maxParticles) {
    // TODO: Is this the correct way to return the particle buffer?
    // ------- SIMULATION PARAMETERS ------- 
    // (m) jet diameter
    const vpmfloat d{ 45.4e-3f };
    vpmfloat U1 = 40.0f;
    // (m/s) Coflow velocity
    vpmfloat U2 = 0.0f; 
    // (deg) Coflow angle from centerline
    vpmvec3 U2angle= vpmvec3 {0.0f};
    //  Ratio of inflow momentum thickness of shear layer to diameter, θ/d
    vpmfloat thetaod = 0.025;
    // Maximum sigmas in z-direction to create annulis for defining BC
    vpmfloat max_zsigma = 12.0f;
    // Threshold at which not to add particles
    vpmfloat minWfraction = 0.01f;
    // Origin of jet
    vpmvec3 jetOrigin = vpmvec3 {0.0f};
    // orientation of jet
    vpmmat3 jetOrientation = vpmmat3 {1.0f};        // Identity matrix

    // -------  SOLVER OPTIONS ------- 
    int steps_per_d = 50;           // Number of time steps for the centerline at U1 to travel one diameter
    int d_travel_tot = 60;          // Run simulation for an equivalent of this many diameters
    vpmfloat maxRoR = 1.0f;            // (m) maximum radial distance to discretize
    vpmfloat dxotheta = 0.25f;        // Distance Δx between particles over momentum thickness θ
    vpmfloat overlap = 2.4f;           // Overlap between particles

    int numParticles{ 0 };

    // Define freestream (coflow) velocity
    vpmvec3 Vfreestream = vpmvec3{ 0, 0, U2 };

    // TODO: How to initialize Uinf = Vinf in pfield

    vpmfloat R = d/2;                         // (m) jet radius
    vpmvec3 Cline = jetOrientation[2];        // Centerline direction

    // Temporal discretization
    vpmfloat dt = d / steps_per_d / U1;         // (s) time step
    int nsteps = static_cast<int>(std::ceil(d_travel_tot * d / U1 / dt));       // Number of time steps

    // Spatial discretization
    vpmfloat maxR       = maxRoR * R;
    vpmfloat dx         = dxotheta * thetaod * d;     // (m) approximate distance between particles
    vpmfloat sigma      = overlap * dx;               // particle smoothing

    // Top-hat velocity profile with smooth edges
    auto Vprofile = [d](vpmfloat r, vpmfloat theta) {
        return std::abs(r) < d / 2 ? std::tanh((d / 2 - std::abs(r)) / theta) : 0.0;
    };
    
    // Vjet lambda function
    auto Vjet = [U1, d, thetaod, Vprofile](vpmfloat r) {
        return U1 * Vprofile(r, thetaod * d);
    };
    
    // -------  SIMULATION SETUP ------- 
    // auto Vjet_wrap = [Vjet](vpmvec3 X){ Vjet(X[1])};
    
    // Convert velocity profile to vorticity profile
    auto dVdr = [d, thetaod, U1](vpmfloat r){
        return U1 * r / (pow(cosh((d - 2 * abs(r))/(2 * thetaod * d)), 2) * thetaod * d * abs(r));
    };

    auto Wr = [dVdr](vpmfloat r){ return -dVdr(r);};
    
    // Brute-force find maximum vorticity in the region to discretize
    int length = 1000;
    vpmfloat step = (2 * maxR)/(length - 1);
    vpmfloat Wpeak = -FLT_MAX;

    for (vpmfloat radius = -maxR; radius <= maxR; radius += step){
        Wpeak = fmax(Wr(radius), Wpeak);
    }
    
    // Number of cross sections
    int Nphi = static_cast<int>(std::ceil(2 * PI * R / dx));
    // Number of radial sections (annuli)
    int NR = static_cast<int>(std::ceil(maxR / dx));
    // (m) actual radial distance between particles
    vpmfloat dr = maxR / NR;

    // Axial component of the freestream
    vpmfloat V2 = glm::dot(Vfreestream, Cline);

    // Boundary condition indices (needed only if we're removing the others before running simul)
    std::vector<int> BCi;
    //Particle * boundaryParticles;

    int startingIndex { 0 };
    // Spatial discretization of the boundary condition
    for (int ri = 1; ri <= NR; ++ri) {      // Iterate over annuli
        
        // Annulus lower, upper bounds and center
        vpmfloat rlo = dr * (ri - 1);
        vpmfloat rup = dr * ri;
        vpmfloat rc = (rlo + rup) / 2;

        // Velocity at center of annulus
        vpmfloat Vc = V2 + Vjet(rc);
        // Distance traveled in one time step
        vpmfloat dz = Vc * dt;

        // Integrate vorticity radially over annulus segment
        // TODO: Confirm: implement closed form solution for -vJet
        vpmfloat Wint = -(Vjet(rup) - Vjet(rlo));

        // Annulus circulation
        vpmfloat circulation = Wint * dz + 1e-12f;
        // Mean vorticity
        vpmfloat Wmean = Wint / (rup - rlo);

        // Area of annulus swept
        vpmfloat area = dz * (rup - rlo);

        // Number of longitudinal divisions
        int Nz = static_cast<int>(std::ceil(max_zsigma * sigma / dz));

        if (abs(Wmean) / Wpeak >= minWfraction) {
            // Iterate over Z layers (time steps)
            for (int zi = 0; zi <= Nz; ++zi) {

                int org_np = numParticles;

                vpmvec3 currentJetOrigin = jetOrigin + zi * dz * Cline;
               
                bool isStatic = zi!=0;
                // Call addAnnulus with appropriate arguments
                startingIndex = addAnnulus(particleBuffer, circulation, R, Nphi, sigma, area,
                            jetOrigin, jetOrientation, isStatic, startingIndex, maxParticles);
                
                if (startingIndex == -1) break;
  
                numParticles = startingIndex;

                // If `zi == 0`, update boundary condition indices
                if (zi == 0) {
                    for (int pi = org_np + 1; pi <= numParticles; ++pi) {
                        BCi.push_back(pi);
                    }
                }

            }
        }
    }   
    int j = 0;
    // BCi always the same in
    for (int i = 0; i < BCi.size(); i++){
        boundaryBuffer[j] = particleBuffer[BCi[i]];
        j++;
    }
    // remove all particles from particleBuffer that are not in the BCi array?
    // need to return initial boundary particle buffer
    return {numParticles, BCi.size()}; // or BCi.size();
}