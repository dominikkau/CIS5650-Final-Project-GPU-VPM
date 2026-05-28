#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "../vpmcore/common.h"
#include "../vpmcore/particlebuffer.h"
#include "fmm_p2m.h"
#include "fmm_m2m.h"

struct HCell
{
	uint64_t index;
	uint64_t morton;
	vpm::vec3 center;
	vpm::real radius;
	uint8_t childMask;
	uint8_t depth;
};

//struct HCell
//{
//	vpm::vec3 center;
//	vpm::real radius;
//	vpm::midx_t morton;
//	vpm::nidx_t index;
//	uint8_t childMask;
//};

struct PEInteraction
{
	size_t expansion;
	size_t pointStart;
	size_t pointEnd;
};

struct Interaction
{
	size_t dest;
	size_t src;
};

struct DomainInfo
{
	vpm::vec3 lower;
	vpm::real size;
};

static DomainInfo calcDomain(const vpm::vec3* points, size_t numPoints);
static std::vector<uint64_t> calcMortonCodes(const vpm::vec3* points, size_t numPoints, const DomainInfo& domain);
static void sortByMorton(ParticleBuffer& particles, size_t numParticles, std::span<uint64_t> mortonCodes);

void testTree();
std::unordered_map<uint64_t, HCell> buildTree(const vpm::vec3* points, size_t numPoints,
	int maxPointsPerNode, const DomainInfo& domain, const std::vector<uint64_t>& mortonCodes,
	P2MInfo& p2mInfo, M2MInfo& m2mInfo);
void dualTreeTraversal(const std::unordered_map<uint64_t, HCell>& treeMap, std::vector<Interaction>& p2pList, std::vector<Interaction>& m2lList);