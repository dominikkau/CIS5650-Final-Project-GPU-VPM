#include <iostream>
#include <cmath>
#include <glm/glm.hpp>
#include "roundjetsimulation.hpp"
#include "vpmcore/kernel.h"

int addAnnulus(Particle* particleBuffer, vpmfloat circulation, vpmfloat R,
    int Nphi, vpmfloat sigma, vpmfloat area, vpmvec3 ringPosition,
    vpmmat3 ringOrientation, int startingIndex, int maxParticles){
        
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
        
        auto fun_X_global = [ringPosition, ringOrientation](vpmvec3 x) {
        return ringPosition + ringOrientation * x;
        };

        auto fun_Gamma_global = [ringOrientation](vpmvec3 Gamma) {
        return ringOrientation * Gamma;
        };

        // Perimeter spacing between cross sections
        vpmfloat dS = Stot / Nphi;
        
        // Non-dimensional perimeter spacing
        vpmfloat ds = dS / Stot;

        int idx = startingIndex;
        // Discretization of annulus into cross-sections
        for (int i = 0; i < Nphi; i++){
            
            // Non-dimensional arc-length position of cross section along centerline
            vpmfloat sc1 = ds * N;        // Lower bound
            vpmfloat sc2 = ds *(N+1);     // Upper bound
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
            ++idx;
        }

    }