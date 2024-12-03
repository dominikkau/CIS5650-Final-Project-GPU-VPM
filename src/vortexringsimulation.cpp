#include <iostream>
#include <cmath>
#include "vpmcore/kernel.h"

// Function to calculate the number of particles
int numberParticles(int Nphi, int nc, int extra_nc = 0) {
    nc = 0;
    int total_layers = nc + extra_nc;
    int layer_sum = 0;
    for (int i = 1; i <= total_layers; ++i) {
        layer_sum += i;
    }
    return Nphi * (1 + 8 * layer_sum);
}

int addVortexRing(ParticleField* pField, float circulation, float R, float Rcross,
                   int Nphi, int nc, float sigmas, int extra_nc, glm::vec3 ringPosition, 
                   glm::mat3 ringOrientation, int startingIndex, int maxParticles) {

    // Lambda function definition
    // Arclength corresponding to phi for circle with radius r
    auto fun_S = [R](float phi) { return R * phi; };
    // Circle circumference
    float Stot = fun_S(2 * M_PI);
    // Non-dimensional arc length from 0 to a given value <=1
    auto fun_s = [fun_S, Stot](float phi) { return fun_S(phi) / Stot; };

    // Angle associated to a given non-dimensional arc length
    auto fun_phi = [R](float s) { return 2 * M_PI * s; }

    auto fun_length = [fun_S, R](float r, float tht, float phi1, float phi2) {
        float S1 = fun_S(phi1, R + r * cos(tht));
        float S2 = fun_S(phi2, R + r * cos(tht));
        return S2 - S1;
    };

    // Volume of a cell in the torus
    auto fun_vol = [R](float phi1, float phi2, float tht1, float tht2, float r1, float r2) {
        float tmp1 = 0.5f * R * (r2 * r2 - r1 * r1) * (tht2 - tht1);
        float tmp2 = (sin(tht2) - sin(tht1)) * (r2 * r2 * r2 - r1 * r1 * r1) / 3.0f;
        return (phi2 - phi1) * (tmp1 + tmp2);
    };

    auto fun_X_global = [ringPosition, ringOrientation](glm::vec3 x) {
        return ringPosition + ringOrientation * x;
    }

    auto fun_Gamma_global = [ringOrientation](glm::vec3 Gamma) {
        return ringOrientation * Gamma;
    }

    float rl = Rcross / (2 * nc + 1);
    float dS = Stot / Nphi;
    float ds = dS / Stot;
    float omega = circulation / (M_PI * Rcross * Rcross);

    int idx = startingIndex;
    for (int N = 0; N < Nphi; ++N) {
        float sc1 = ds * N;
        float sc2 = ds * (N + 1);
        float sc = (sc1 + sc2) / 2;

        float phi1 = fun_phi(sc1);
        float phi2 = fun_phi(sc2);
        float phic = fun_phi(sc);

        glm::vec3 Xc{ R * sin(phic), R * cos(phic), 0 }; // Center of the cross section
        glm::vec3 T{ -cos(phic), sin(phic), 0 }; // Unitary tangent of the cross section
        glm::mat3 Naxis;
        Naxis[0] = T;
        Naxis[1] = glm::cross(glm::vec3(0, 0, 1), T);
        Naxis[2] = glm::vec3(0, 0, 1);

        for (int n = 0; n <= nc + extra_nc; ++n) {
            if (n == 0) {
                // Compute volume
                float vol = fun_vol(ph1, phi2, 0.0f, 2.0f * M_PI, 0.0f, rl);
                // Position
                glm::vec3 X = Xc;
                // Vortex strength
                glm::vec3 Gamma = omega * vol * T;
                // Filament length
                float length = fun_length(0, R, phi1, phi2);
                // Circulation
                float crcltn = glm::length(Gamma) / length;

                if (idx + 1 >= maxParticles) return -1;

                pField->particles[idx].X = fun_X_global(X);
                pField->particles[idx].Gamma = fun_Gamma_global(Gamma);
                pField->particles[idx].circulation = crcltn;
                ++idx;
            }
            else {
                float rc = (1 + 12 * n * n) / (6 * n) * rl;  // Center radius
                float r1 = (2 * n - 1) * rl;                // Lower radius
                float r2 = (2 * n + 1) * rl;                // Upper radius
                int ncells = 8 * n;                         // Number of cells
                float deltatheta = 2 * M_PI / ncells;       // Angle of cells

                // Discretize layer into cells around the circumference
                for (int j = 0; j < ncells; ++j) {
                    float tht1 = deltatheta * j;            // Left angle
                    float tht2 = deltatheta * (j + 1);      // Right angle
                    float thtc = (tht1 + tht2) / 2;         // Center angle

                    float vol = fun_vol(ph1, ph2, tht1, tht2, r1, r2); // Volume

                    glm::vec3 X = Xc + Naxis * glm::vec3{ 0, rc*cos(thtc), rc*sin(thtc) };

                    glm::vec3 temp = glm::vec3(0, rc*std::cos(thtc), rc*std::sin(thtc));
                    pField->particles[idx].X = Xc + Naxis * temp;

                    glm::vec3 Gamma = (n <= nc) ? omega * vol * T : EPS * T;
                    // Filament length
                    float length = fun_length(0, R, phi1, phi2);
                    // Circulation
                    float crcltn = glm::length(Gamma) / length;

                    if (idx + 1 >= maxParticles) return -1;

                    pField->particles[idx].X = fun_X_global(X);
                    pField->particles[idx].Gamma = fun_Gamma_global(Gamma);
                    pField->particles[idx].circulation = crcltn;
                    ++idx;
                }   
            }   
        }
    }

    return idx;
}

int initVortexRings(Particle* particleBuffer, int maxParticles) {
    int nrings{ 2 };
    float dZ{ 0.7906f };
    int numParticles{ 0 };

    float circulations[nrings];
    float Rs[nrings];
    float Rcrosss[nrings];
    float sigmas[nrings];
    int Nphis[nrings];
    int ncs[nrings];
    int extra_ncs[nrings];
    glm::vec3 ringPositions[nrings];
    glm::mat3 ringOrientations[nrings];

    for (int i = 0; i < nrings; ++i) {
        circulations[i] = 1.0f;
        Rs[i] = 0.7906f;
        Rcrosss[i] = 0.07906f;
        sigmas[i] = 0.07906f;
        Nphis[i] = 100;
        ncs[i] = 1;
        extra_ncs[i] = 0;
        ringPositions[i] = glm::vec3{ 0, 0, dZ * i };
        ringOrientations[i] = glm::mat3{ 1.0f };

        numParticles += numberParticles(Nphis[i], ncs[i], extra_ncs[i]);
    }

    if (numParticles > maxParticles) {
        numParticles = maxParticles;
        std::cout << "Number of particles (" << numParticles;
        std::cout << ") exceeds particleBuffer size (" << maxParticles << ")!" << std::endl;
    }

    int startingIndex{ 0 };
    for (int i = 0; i < nrings; ++i) {
        startingIndex = addVortexRing(particleBuffer, circulations[i], Rs[i], Rcross[i],
                      Nphis[i], ncs[i], sigmas[i], extra_ncs[i], ringPositions[i], 
                      ringOrientations[i], startingIndex, maxParticles);

        if (startingIndex == -1) break;
    }

    return numParticles;
}