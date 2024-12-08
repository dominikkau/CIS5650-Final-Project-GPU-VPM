#include <iostream>
#include <cmath>
#include <glm/glm.hpp>
#include "roundjetsimulation.hpp"
#include "vpmcore/kernel.h"


// Function to calculate the rotation matrix
glm::mat3 rotation_matrix2(vpmvec3 U2angle) {
    // Convert angles from degrees to radians
    vpmfloat a = glm::radians(U2angle.z);   // Yaw
    vpmfloat b = glm::radians(U2angle.y); // Pitch
    vpmfloat g = glm::radians(U2angle.x);  // Roll

    // Rotation matrix about Z-axis (Yaw)
    glm::mat3 Rz = {
        glm::vec3(cos(a), -sin(a), 0.0f),
        glm::vec3(sin(a),  cos(a), 0.0f),
        glm::vec3(0.0f,    0.0f,   1.0f)
    };

    // Rotation matrix about Y-axis (Pitch)
    glm::mat3 Ry = {
        glm::vec3(cos(b),  0.0f, sin(b)),
        glm::vec3(0.0f,    1.0f, 0.0f),
        glm::vec3(-sin(b), 0.0f, cos(b))
    };

    // Rotation matrix about X-axis (Roll)
    glm::mat3 Rx = {
        glm::vec3(1.0f, 0.0f,    0.0f),
        glm::vec3(0.0f, cos(g), -sin(g)),
        glm::vec3(0.0f, sin(g),  cos(g))
    };

    // Combined rotation matrix
    return Rz * Ry * Rx;
}


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


int initRoundJet(Particle * particleBuffer, int maxParticles){
    
    // ------- SIMULATION PARAMETERS ------- 
    vpmfloat thetaod=0.025;         // Ratio of inflow momentum thickness of shear layer to diameter, θ/d

    // (m) jet diameter
    const vpmfloat d{ 1.5812f };
    vpmfloat U1;
    // (m/s) Coflow velocity
    vpmfloat U2; 
    // (deg) Coflow angle from centerline
    vpmvec3 U2angle= vpmvec3 {0.0f};
    // Origin of jet
    vpmvec3 jetOrigin = vpmvec3 {0.0f};
    // orientation of jet
    vpmmat3 jetOrientation = vpmmat3 {1.0f};        // Identity matrix

    // -------  SOLVER OPTIONS ------- 
    int steps_per_d = 50;           // Number of time steps for the centerline at U1 to travel one diameter
    int d_travel_tot = 10;          // Run simulation for an equivalent of this many diameters
    vpmfloat maxRoR=1.0;            // (m) maximum radial distance to discretize
    vpmfloat dxotheta = 1/4;        // Distance Δx between particles over momentum thickness θ
    vpmfloat overlap=2.4;           // Overlap between particles

    int numParticles{ 0 };

    // Define freestream (coflow) velocity
    vpmvec3 Vfreestream = U2 * (rotation_matrix2(U2angle) * jetOrientation[2]);

    // TODO: How to initialize Uinf = Vinf in pfield
    // TODO: Calculate the integral Wint and initialize circulations using it

    vpmfloat R = d/2;                         // (m) jet radius
    vpmvec3 Cline = jetOrientation[2];        // Centerline direction

    // Temporal discretization
    vpmfloat dt = d / steps_per_d / U1;         // (s) time step
    int nsteps = static_cast<int>(std::ceil(d_travel_tot * d / U1 / dt));       // Number of time steps

    // Spatial discretization
    vpmfloat maxR       = maxRoR * R;
    vpmfloat dx         = dxotheta * thetaod * d;     // (m) approximate distance between particles
    vpmfloat sigma      = overlap * dx;               // particle smoothing

    // Velocity profile lambda function
    auto Vprofile = [](double r, double d, double theta) {
        return std::abs(r) < d / 2 ? std::tanh((d / 2 - std::abs(r)) / theta) : 0.0;
    };
    
    // Vjet lambda function
    auto Vjet = [U1, d, thetaod, Vprofile](vpmfloat r) {
        return U1 * Vprofile(r, d, thetaod * d);
    };
    
    // -------  SIMULATION SETUP ------- 
    auto Vjet_wrap = [Vjet](vpmvec3 X){ Vjet(X[1])};

    // TODO: gradient of Vjet_wrap wrt X
    vpmfloat sigma;
    int Nphi;
}