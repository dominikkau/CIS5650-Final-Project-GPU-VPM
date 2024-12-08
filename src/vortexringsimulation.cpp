#include <iostream>
#include <cmath>
#include <vector>
#include <glm/glm.hpp>
#include "vortexringsimulation.hpp"
#include "vpmcore/kernel.h"

// Function to calculate the number of particles
int numberParticles(int Nphi, int nc, int extra_nc) {
    return Nphi * (1 + 4 * (nc + extra_nc) * (nc + extra_nc + 1));
}

int addVortexRing(Particle* particleBuffer, vpmfloat circulation, vpmfloat R, vpmfloat Rcross,
                   int Nphi, int nc, vpmfloat sigma, int extra_nc, vpmvec3 ringPosition, 
                   vpmmat3 ringOrientation, int startingIndex, int maxParticles) {
    // Lambda function definition
    // Arclength corresponding to phi for circle with radius r
    auto fun_S = [](vpmfloat phi, vpmfloat r) { return r * phi; };
    // Circle circumference
    vpmfloat Stot = fun_S(2 * PI, R);
    // Non-dimensional arc length from 0 to a given value <=1
    auto fun_s = [fun_S, Stot](vpmfloat phi, vpmfloat r) { return fun_S(phi, r) / Stot; };

    // Angle associated to a given non-dimensional arc length
    auto fun_phi = [](vpmfloat s) { return 2 * PI * s; };

    auto fun_length = [fun_S, R](vpmfloat r, vpmfloat tht, vpmfloat phi1, vpmfloat phi2) {
        vpmfloat S1 = fun_S(phi1, R + r * cos(tht));
        vpmfloat S2 = fun_S(phi2, R + r * cos(tht));
        return S2 - S1;
    };

    // Volume of a cell in the torus
    auto fun_vol = [R](vpmfloat phi1, vpmfloat phi2, vpmfloat tht1, vpmfloat tht2, vpmfloat r1, vpmfloat r2) {
        vpmfloat tmp1 = 0.5f * R * (r2 * r2 - r1 * r1) * (tht2 - tht1);
        vpmfloat tmp2 = (sin(tht2) - sin(tht1)) * (r2 * r2 * r2 - r1 * r1 * r1) / 3.0f;
        return (phi2 - phi1) * (tmp1 + tmp2);
        };

    auto fun_X_global = [ringPosition, ringOrientation](vpmvec3 x) {
        return ringPosition + ringOrientation * x;
        };

        auto fun_Gamma_global = [ringOrientation](vpmvec3 Gamma) {
        return ringOrientation * Gamma;
        };

    vpmfloat rl = Rcross / (2 * nc + 1);
    vpmfloat dS = Stot / Nphi;
    vpmfloat ds = dS / Stot;
    vpmfloat omega = circulation / (PI * Rcross * Rcross);

    int idx = startingIndex;
    for (int N = 0; N < Nphi; ++N) {
        vpmfloat sc1 = ds * N;
        vpmfloat sc2 = ds * (N + 1);
        vpmfloat sc = (sc1 + sc2) / 2;

        vpmfloat phi1 = fun_phi(sc1);
        vpmfloat phi2 = fun_phi(sc2);
        vpmfloat phic = fun_phi(sc);

        vpmvec3 Xc{ R * sin(phic), R * cos(phic), 0 }; // Center of the cross section
        vpmvec3 T{ -cos(phic), sin(phic), 0 }; // Unitary tangent of the cross section
        vpmmat3 Naxis;
        Naxis[0] = T;
        Naxis[1] = glm::cross(vpmvec3(0, 0, 1), T);
        Naxis[2] = vpmvec3(0, 0, 1);

        for (int n = 0; n <= nc + extra_nc; ++n) {
            if (n == 0) {
                // Compute volume
                vpmfloat vol = fun_vol(phi1, phi2, 0.0f, 2.0f * PI, 0.0f, rl);
                // Position
                vpmvec3 X = Xc;
                // Vortex strength
                vpmvec3 Gamma = omega * vol * T;
                // Filament length
                vpmfloat length = fun_length(0, R, phi1, phi2);
                // Circulation
                vpmfloat crcltn = glm::length(Gamma) / length;

                if (idx >= maxParticles - 1) return -1;

                particleBuffer[idx].X = fun_X_global(X);
                particleBuffer[idx].Gamma = fun_Gamma_global(Gamma);
                particleBuffer[idx].circulation = crcltn;
                particleBuffer[idx].sigma = sigma;
                particleBuffer[idx].vol = vol;
                particleBuffer[idx].index = idx;
                ++idx;
            }
            else {
                vpmfloat rc = (1 + 12 * n * n) / (6 * n) * rl;  // Center radius
                vpmfloat r1 = (2 * n - 1) * rl;                // Lower radius
                vpmfloat r2 = (2 * n + 1) * rl;                // Upper radius
                int ncells = 8 * n;                         // Number of cells
                vpmfloat deltatheta = 2 * PI / ncells;       // Angle of cells

                // Discretize layer into cells around the circumference
                for (int j = 0; j < ncells; ++j) {
                    vpmfloat tht1 = deltatheta * j;            // Left angle
                    vpmfloat tht2 = deltatheta * (j + 1);      // Right angle
                    vpmfloat thtc = (tht1 + tht2) / 2;         // Center angle

                    vpmfloat vol = fun_vol(phi1, phi2, tht1, tht2, r1, r2); // Volume

                    vpmvec3 X = Xc + Naxis * vpmvec3{ 0, rc*cos(thtc), rc*sin(thtc) };

                    vpmvec3 Gamma = (n <= nc) ? omega * vol * T : EPS * T;
                    // Filament length
                    vpmfloat length = fun_length(0, R, phi1, phi2);
                    // Circulation
                    vpmfloat crcltn = glm::length(Gamma) / length;

                    if (idx >= maxParticles - 1) return -1;

                    particleBuffer[idx].X = fun_X_global(X);
                    particleBuffer[idx].Gamma = fun_Gamma_global(Gamma);
                    particleBuffer[idx].circulation = crcltn;
                    particleBuffer[idx].sigma = sigma;
                    particleBuffer[idx].vol = vol;
                    particleBuffer[idx].index = idx;
                    ++idx;
                }   
            }   
        }
    }

    return idx;
}

int initVortexRings(Particle* particleBuffer, int maxParticles) {
    // Number of rings
    const int nrings{ 2 };
    // Offset of rings
    vpmfloat dZ{ 0.7906f };

    int numParticles{ 0 };
    vpmfloat circulations[nrings];
    vpmfloat Rs[nrings];
    vpmfloat Rcrosss[nrings];
    vpmfloat sigmas[nrings];
    int Nphis[nrings];
    int ncs[nrings];
    int extra_ncs[nrings];
    vpmvec3 ringPositions[nrings];
    vpmmat3 ringOrientations[nrings];

    for (int i = 0; i < nrings; ++i) {
        circulations[i] = 1.0f;
        Rs[i] = 0.7906f;
        Rcrosss[i] = 0.07906f;
        sigmas[i] = 0.07906f;
        Nphis[i] = 100;
        ncs[i] = 1;
        extra_ncs[i] = 0;
        ringPositions[i] = vpmvec3{ 0, 0, dZ * i };
        ringOrientations[i] = vpmmat3{ 1.0f };

        numParticles += numberParticles(Nphis[i], ncs[i], extra_ncs[i]);
    }

    if (numParticles > maxParticles) {
        std::cout << "Number of particles (" << numParticles;
        std::cout << ") exceeds particleBuffer size (" << maxParticles << ")!" << std::endl;
        numParticles = maxParticles;
    }

    int startingIndex{ 0 };
    for (int i = 0; i < nrings; ++i) {
        startingIndex = addVortexRing(particleBuffer, circulations[i], Rs[i], Rcrosss[i],
                      Nphis[i], ncs[i], sigmas[i], extra_ncs[i], ringPositions[i], 
                      ringOrientations[i], startingIndex, maxParticles);

        if (startingIndex == -1) break;
    }

    return numParticles;
}