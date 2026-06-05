#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "../vpmcore/common.h"
#include "../vpmcore/particlebuffer.h"
#include "fmm_p2m.h"
#include "fmm_m2m.h"
#include "fmm_m2l.h"

struct HCell
{
	vpm::midx_t morton;
	vpm::vec3 center;
	vpm::real radius;
	vpm::nidx_t index;
	uint8_t childMask;
	uint8_t depth;
};

struct DomainInfo
{
	vpm::vec3 lower;
	vpm::real size;
};

struct Interaction
{
	vpm::nidx_t dest;
	vpm::nidx_t src;
};

static DomainInfo calcDomain(const vpm::vec3* points, vpm::pidx_t numPoints);
static std::vector<vpm::midx_t> calcMortonCodes(const vpm::vec3* points, vpm::pidx_t numPoints, const DomainInfo& domain);
static void sortByMorton(ParticleBuffer& particles, vpm::pidx_t numParticles, std::span<vpm::midx_t> mortonCodes);

void testTree();
std::unordered_map<vpm::midx_t, HCell> buildTree(const vpm::vec3* points, vpm::pidx_t numPoints,
	int maxPointsPerNode, const DomainInfo& domain, const std::vector<vpm::midx_t>& mortonCodes,
	P2MInfo& p2mInfo, M2MInfo& m2mInfo);
void dualTreeTraversal(const std::unordered_map<vpm::midx_t, HCell>& treeMap, std::vector<Interaction>& p2pList, M2LInfo& m2lInfo);