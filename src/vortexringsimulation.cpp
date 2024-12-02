#include <iostream>
#include <stdlib.h>
#include "vpmcore/subFilterScale.h"
#include "vpmcore/relaxation.h"
#include "vpmcore/kernels.h"
#include "vpmcore/timeIntegration.h"
#include "vpmcore/particleField.h"

using namespace std;

int nsteps = 100;
float Rtot = 2.0f;
int nrings = 1;
float dZ = 0.1f;

vector<float> circulations(nrings, 1.0);
vector<float> Rs(nrings, 1.0);
vector<float> ARs(nrings, 1.0);
vector<float> Rcrosss(nrings, 0.15 * Rs[0]);
vector<float> sigmas = Rcrosss;
vector<int> Nphis(nrings, 100);
vector<int> ncs(nrings, 0); // CHECK
vector<int> extra_ncs(nrings, 0);
vector<vector<float>> ringPosition;

for(int ri = 0; ri < nrings; ++ri) {
    ringPosition.push_back({0.0f, 0.0f, dZ * (ri)});
}
vector<vector<float>> ringOrientation(nrings, {1.0f, 1.0f, 1.0f}); // Placeholder for orientation

int nref = 1;
float beta = 0.5f;
float faux = 0.25f;

// solver settings
ParticleField* pField =  run_vortexring_simulation(
            nrings, circulations, Rs, ARs, Rcrosss,
            Nphis, ncs, extra_ncs, ringPosition, ringOrientation,
            nref, nsteps, Rtot, beta, faux, NoSFS, PedrizzettiRelaxation, 
            WinckelmansKernel, true
        );
