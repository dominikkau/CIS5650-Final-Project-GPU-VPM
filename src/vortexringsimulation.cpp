#include <iostream>
#include <cmath>
#include <vector>
#include <glm/glm.hpp>
#include "vortexringsimulation.hpp"
#include "vpmcore/vpmmain.h"

// Function to calculate the number of particles
vpm::pidx_t vortex_rings::numberParticles(int Nphi, int nc, int extra_nc) {
    return Nphi * (1 + 4 * (nc + extra_nc) * (nc + extra_nc + 1));
}

// Function to calculate the number of particles
vpm::pidx_t vortex_rings::numberParticles(const VortexRing &ring) {
    return ring.Nphi * (1 + 4 * (ring.nc + ring.extra_nc) * (ring.nc + ring.extra_nc + 1));
}

// Function to calculate the number of particles
vpm::pidx_t vortex_rings::numberParticles(const std::vector<VortexRing>& rings) {
    vpm::pidx_t totalParticles = 0;
	for (const auto& ring : rings)
		totalParticles += numberParticles(ring);
	return totalParticles;
}

vpm::pidx_t vortex_rings::addVortexRing(ParticleBuffer& particleBuffer, vpm::real circulation, vpm::real R, vpm::real Rcross,
    int Nphi, int nc, vpm::real sigma, int extra_nc, vpm::vec3 ringPosition,
    vpm::mat3 ringOrientation, vpm::pidx_t startingIndex) {
    // Lambda function definition
    // Arclength corresponding to phi for circle with radius r
    auto fun_S = [](vpm::real phi, vpm::real r) { return r * phi; };
    // Circle circumference
    vpm::real Stot = fun_S(2 * PI, R);
    // Non-dimensional arc length from 0 to a given value <=1
    auto fun_s = [fun_S, Stot](vpm::real phi, vpm::real r) { return fun_S(phi, r) / Stot; };

    // Angle associated to a given non-dimensional arc length
    auto fun_phi = [](vpm::real s) { return 2 * PI * s; };

    auto fun_length = [fun_S, R](vpm::real r, vpm::real tht, vpm::real phi1, vpm::real phi2) {
        vpm::real S1 = fun_S(phi1, R + r * cos(tht));
        vpm::real S2 = fun_S(phi2, R + r * cos(tht));
        return S2 - S1;
        };

    // Volume of a cell in the torus
    auto fun_vol = [R](vpm::real phi1, vpm::real phi2, vpm::real tht1, vpm::real tht2, vpm::real r1, vpm::real r2) {
        vpm::real tmp1 = 0.5f * R * (r2 * r2 - r1 * r1) * (tht2 - tht1);
        vpm::real tmp2 = (sin(tht2) - sin(tht1)) * (r2 * r2 * r2 - r1 * r1 * r1) / 3.0f;
        return (phi2 - phi1) * (tmp1 + tmp2);
        };

    auto fun_X_global = [ringPosition, ringOrientation](vpm::vec3 x) {
        return ringPosition + ringOrientation * x;
        };

    auto fun_Gamma_global = [ringOrientation](vpm::vec3 Gamma) {
        return ringOrientation * Gamma;
        };

    vpm::real rl = Rcross / (2 * nc + 1);
    vpm::real dS = Stot / Nphi;
    vpm::real ds = dS / Stot;
    vpm::real omega = circulation / (PI * Rcross * Rcross);

    vpm::pidx_t idx = startingIndex;
    for (int N = 0; N < Nphi; ++N) {
        vpm::real sc1 = ds * N;
        vpm::real sc2 = ds * (N + 1);
        vpm::real sc = (sc1 + sc2) / 2;

        vpm::real phi1 = fun_phi(sc1);
        vpm::real phi2 = fun_phi(sc2);
        vpm::real phic = fun_phi(sc);

        vpm::vec3 Xc{ R * sin(phic), R * cos(phic), 0 }; // Center of the cross section
        vpm::vec3 T{ -cos(phic), sin(phic), 0 }; // Unitary tangent of the cross section
        vpm::mat3 Naxis;
        Naxis[0] = T;
        Naxis[1] = glm::cross(vpm::vec3(0, 0, 1), T);
        Naxis[2] = vpm::vec3(0, 0, 1);

        for (int n = 0; n <= nc + extra_nc; ++n) {
            if (n == 0) {
                // Compute volume
                vpm::real vol = fun_vol(phi1, phi2, 0.0f, 2.0f * PI, 0.0f, rl);
                // Position
                vpm::vec3 X = Xc;
                // Vortex strength
                vpm::vec3 Gamma = omega * vol * T;
                // Filament length
                vpm::real length = fun_length(0, R, phi1, phi2);
                // Circulation
                vpm::real crcltn = glm::length(Gamma) / length;

                if (idx >= particleBuffer.size()) return 0;

                particleBuffer.X()[idx] = fun_X_global(X);
                particleBuffer.Gamma()[idx] = fun_Gamma_global(Gamma);
                //particleBuffer.circulation()[idx] = crcltn;
                particleBuffer.sigma()[idx] = sigma;
                //particleBuffer.vol()[idx] = vol;
                particleBuffer.index()[idx] = idx;
                ++idx;
            }
            else {
                vpm::real rc = (1 + 12 * n * n) / (6 * n) * rl;  // Center radius
                vpm::real r1 = (2 * n - 1) * rl;                // Lower radius
                vpm::real r2 = (2 * n + 1) * rl;                // Upper radius
                int ncells = 8 * n;                         // Number of cells
                vpm::real deltatheta = 2 * PI / ncells;       // Angle of cells

                // Discretize layer into cells around the circumference
                for (int j = 0; j < ncells; ++j) {
                    vpm::real tht1 = deltatheta * j;            // Left angle
                    vpm::real tht2 = deltatheta * (j + 1);      // Right angle
                    vpm::real thtc = (tht1 + tht2) / 2;         // Center angle

                    vpm::real vol = fun_vol(phi1, phi2, tht1, tht2, r1, r2); // Volume

                    vpm::vec3 X = Xc + Naxis * vpm::vec3{ 0, rc * cos(thtc), rc * sin(thtc) };

                    vpm::vec3 Gamma = (n <= nc) ? omega * vol * T : EPS * T;
                    // Filament length
                    vpm::real length = fun_length(0, R, phi1, phi2);
                    // Circulation
                    vpm::real crcltn = glm::length(Gamma) / length;

                    if (idx >= particleBuffer.size()) return 0;

                    particleBuffer.X()[idx] = fun_X_global(X);
                    particleBuffer.Gamma()[idx] = fun_Gamma_global(Gamma);
                    //particleBuffer.circulation()[idx] = crcltn;
                    particleBuffer.sigma()[idx] = sigma;
                    //particleBuffer.vol()[idx] = vol;
                    particleBuffer.index()[idx] = idx;
                    ++idx;
                }
            }
        }
    }

    return idx;
}

vpm::pidx_t vortex_rings::initVortexRings(ParticleBuffer& particleBuffer) {
    // Number of rings
    const int nrings{ 2 };
    // Offset of rings
    vpm::real dZ{ 0.7906f };

    vpm::pidx_t numParticles{ 0 };
    vpm::real circulations[nrings];
    vpm::real Rs[nrings];
    vpm::real Rcrosss[nrings];
    vpm::real sigmas[nrings];
    int Nphis[nrings];
    int ncs[nrings];
    int extra_ncs[nrings];
    vpm::vec3 ringPositions[nrings];
    vpm::mat3 ringOrientations[nrings];

    for (int i = 0; i < nrings; ++i) {
        circulations[i] = 1.0f;
        Rs[i] = 0.7906f;
        Rcrosss[i] = 0.07906f;
        sigmas[i] = 0.07906f;
        Nphis[i] = 200;
        ncs[i] = 3;
        extra_ncs[i] = 0;
        ringPositions[i] = vpm::vec3{ 0, 0, dZ * i };
        ringOrientations[i] = vpm::mat3{ 1.0f };

        numParticles += numberParticles(Nphis[i], ncs[i], extra_ncs[i]);
    }

    if (numParticles > particleBuffer.size()) {
        std::cout << "Number of particles (" << numParticles;
        std::cout << ") exceeds particleBuffer size (" << particleBuffer.size() << ")!" << std::endl;
        numParticles = particleBuffer.size();
    }

    vpm::pidx_t startingIndex{ 0 };
    for (int i = 0; i < nrings; ++i) {
        startingIndex = addVortexRing(particleBuffer, circulations[i], Rs[i], Rcrosss[i],
            Nphis[i], ncs[i], sigmas[i], extra_ncs[i], ringPositions[i],
            ringOrientations[i], startingIndex);

        if (startingIndex == 0) break;
    }

    return numParticles;
}

vpm::pidx_t vortex_rings::initParticleBuffer(ParticleBuffer& particleBuffer, const std::vector<VortexRing> &rings)
{
    vpm::pidx_t numParticles = numberParticles(rings);
    if (numParticles > particleBuffer.size()) {
        std::cout << "Number of particles (" << numParticles;
        std::cout << ") exceeds particleBuffer size (" << particleBuffer.size() << ")!" << std::endl;
        numParticles = particleBuffer.size();
    }

    vpm::pidx_t startingIndex{ 0 };
	for (const auto &ring: rings) {
		startingIndex = addVortexRing(particleBuffer, ring.circulation, ring.R, ring.Rcross,
			ring.Nphi, ring.nc, ring.sigma, ring.extra_nc, ring.position,
			ring.orientation, startingIndex);

        if (startingIndex == 0) break;
	}

    return numParticles;
}