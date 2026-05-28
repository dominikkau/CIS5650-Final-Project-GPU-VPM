#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <execution>
#include <iostream>
#include <bitset>
#include <random>
#include <chrono>
#include <numeric>
#include <array>
#include "tree.h"
#include "fmm_utils.h"
#include "../vpmcore/vpmmain.h"
#include "../vortexringsimulation.hpp"
#include "../lean_vtk.hpp"
#include "../vpmio.h"

#ifdef __INTELLISENSE__
#define __CUDACC__
#endif // __INTELLISENSE__

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__


static constexpr uint8_t CHILD_MASK = 0b111;
static constexpr vpm::real MAC_COEFFICIENT = 0.76371f;
static constexpr vpm::real DOMAIN_EPSILON = 1e-6f;

void testTree()
{
	//// Define vortex rings properties
 //   std::vector<vortex_rings::VortexRing> vortexRings(4);
 //   const float dz = 0.7906f;
	//for (int i = 0; i < vortexRings.size(); i++)
	//{
 //       vortexRings[i].circulation = 1.0f;
 //       vortexRings[i].R = 0.7906f;
 //       vortexRings[i].Rcross = 0.07906f;
 //       vortexRings[i].sigma = 0.07906f;
 //       vortexRings[i].Nphi = 200;
 //       vortexRings[i].nc = 3;
 //       vortexRings[i].position = { 0, 0, dz * i };
	//}

	//size_t numParticles = vortex_rings::numberParticles(vortexRings);
	//std::cout << "Total number of particles: " << numParticles << std::endl;

	//// Compute vortex ring particle positions and strengths
 //   ParticleBuffer particleBuffer{ ParticleBufferType::HOST, numParticles };
 //   int inputBufferMask = BufferField::X | BufferField::INDEX | BufferField::GAMMA | BufferField::SIGMA;
 //   particleBuffer.mallocFields(inputBufferMask);

	//size_t numParticlesTrue = vortex_rings::initParticleBuffer(particleBuffer, vortexRings);
	//std::cout << "Initialized " << numParticlesTrue << " particles in buffer" << std::endl;

	constexpr int REPETITIONS = 10;
	size_t numParticles = 10'000;
	// Degree of expansion
	constexpr int p = 8;

	fmm::writeSwapCoefs(p);

	std::random_device rd;
	std::mt19937 gen(13); // Mersenne Twister generator
	std::normal_distribution<> disPos(0.0, 1.0); // mean=0, std_dev=1
	std::normal_distribution<> disGamma(-1.0, 1.0); // mean=0, std_dev=1
	//std::uniform_real_distribution<> dis(0.0, 1.0);

	std::vector<vpm::vec3> positions;

	ParticleBuffer particles{ ParticleBufferType::HOST, numParticles };
	int inputBufferMask = BufferField::X | BufferField::INDEX | BufferField::GAMMA;
	particles.mallocFields(inputBufferMask);
	for (size_t i = 0; i < numParticles; ++i) {
		particles.X()[i].x = disPos(gen);
		particles.X()[i].y = disPos(gen);
		particles.X()[i].z = disPos(gen);
		particles.Gamma()[i].x = disGamma(gen);
	}

	std::cout << "Allocated and initialized particle buffer with " << numParticles << " particles." << std::endl;

	//vpmio::writeParticleBuffer("../output/verification_test.txt", particles, numParticles, BufferField::X | BufferField::GAMMA);

	// Set up particle buffer on device and copy data
	ParticleBuffer dev_particles{ ParticleBufferType::DEVICE, numParticles };
	dev_particles.mallocFields(inputBufferMask);
	checkCUDAError("Memory allocation for particle field failed");
	cpyParticleBuffer(dev_particles, particles, inputBufferMask);
	checkCUDAError("Copy of particle field failed");

	// Compute domain limits
	DomainInfo domain = calcDomain(particles.X(), numParticles);

	// Compute morton codes for all points
	std::vector<vpm::midx_t> mortonCodes = calcMortonCodes(particles.X(), numParticles, domain);

	// Sort particle buffer by morton codes
	sortByMorton(particles, numParticles, mortonCodes);

	// Build adaptive Octree
	int maxParticlesPerNode = 50;
	//std::vector<PEInteraction> p2mList;
	//p2mList.reserve(numParticles / maxParticlesPerNode * 4); // Rough estimate for number of leaf nodes
	P2MInfo p2mInfo{ numParticles / maxParticlesPerNode * 4 };
	M2MInfo m2mInfo{ };
	auto treeMap = buildTree(particles.X(), numParticles, maxParticlesPerNode, domain, mortonCodes, p2mInfo, m2mInfo);

	p2mInfo.toDevice();
	m2mInfo.toDevice();

	const size_t nodeCount = treeMap.size();
	float* dev_M = nullptr;
	cudaMalloc(&dev_M, nodeCount * p * p * sizeof(float));
	checkCUDAError("Memory of expansion coefficients failed");

	cudaMemset(dev_M, 0, nodeCount * p * p * sizeof(float));

	std::cout << "Total number of nodes in tree: " << nodeCount << std::endl;
	std::cout << "Total number of P2M interactions: " << p2mInfo.size() << std::endl;
	for (int i = 0; i < m2mInfo.depthCount(); ++i) {
		std::cout << "Depth " << i + 1 << ": " << m2mInfo.size(i) << " M2M interactions" << std::endl;
	}
	//std::cout << "Total number of M2M interactions: " << m2mInfo.size() << std::endl;
	
	//size_t* dev_idxList = nullptr;
	//cudaMalloc(&dev_idxList, nodeCount * 4 * sizeof(size_t));
	//checkCUDAError("Memory allocation for indices failed");

	//cudaMemcpy(dev_idxList, reinterpret_cast<size_t*>(p2mList.data()), p2mList.size() * sizeof(PEInteraction), cudaMemcpyHostToDevice);
	//checkCUDAError("Copy of indices failed");

	



	constexpr unsigned int blockSize = 64;
	const unsigned int numBlocks = (p2mInfo.size() + blockSize - 1) / blockSize;

	const unsigned int sharedMemSize = blockSize * (sizeof(float) * p * p);

	std::cout << "Shared memory: " << sharedMemSize << " bytes per block" << std::endl;


    // Create CUDA events
    cudaEvent_t cudaStart, cudaStop;
    cudaEventCreate(&cudaStart);
    cudaEventCreate(&cudaStop);

    // Record the start event
    cudaEventRecord(cudaStart);

#pragma unroll
	for (int i = 0; i < REPETITIONS; ++i) {
		//fmmP2M << <numBlocks, blockSize, sharedMemSize >> > (p2mInfo.dev_nodes(), p2mInfo.dev_pointsStart(), p2mInfo.dev_pointsEnd(), p2mInfo.dev_centers(), p2mInfo.size(), dev_particles.X(), dev_particles.Gamma(), dev_M, p);
		fmm::p2m<<<numBlocks, blockSize, sharedMemSize>>>(p2mInfo.dev_nodes(), p2mInfo.dev_pointsEnd(), p2mInfo.dev_centers(), p2mInfo.size(), dev_particles.X(), dev_particles.Gamma(), dev_M, p);
		checkCUDAError("Kernel fmmP2M failed");
		cudaDeviceSynchronize();
	}

	// Record the stop event
    cudaEventRecord(cudaStop);
    cudaEventSynchronize(cudaStop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, cudaStart, cudaStop);

	cudaEventDestroy(cudaStart);
	cudaEventDestroy(cudaStop);

    // Output the duration
    std::cout << "Kernel execution took " << milliseconds / REPETITIONS << " ms on average" << std::endl;

	constexpr int checkCoeffs = 10;
	float* expansionCoeffs = new float[nodeCount * p * p];

	cudaMemcpy(expansionCoeffs, dev_M, nodeCount * p * p * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("Copy of indices failed");

	std::cout << "First " << checkCoeffs << " expansion coefficients (M) for first " << checkCoeffs << " leaf nodes:" << std::endl;
	for (int i = 0; i < checkCoeffs; ++i) {
		size_t nodeIndex = p2mInfo.nodes()[i];
		std::cout << "Node " << nodeIndex << ": ";
		for (int j = 0; j < 5; ++j) {
			std::cout << expansionCoeffs[nodeIndex * p * p + j] << " ";
		}
		std::cout << std::endl;
	}


	int maxDepth = m2mInfo.depthCount();
	constexpr unsigned int blockSizeM2M = 64;
	const unsigned int numBlocksM2M = (m2mInfo.size(maxDepth-1) + blockSizeM2M - 1) / blockSizeM2M;

	const unsigned int sharedMemSizeM2M = blockSizeM2M * (p * p + 2 * p - 1 + (p % 2 == 0)) * sizeof(float);

	
	fmm::m2m<<<numBlocksM2M, blockSizeM2M, sharedMemSizeM2M>>>(m2mInfo.dev_parents(maxDepth - 1), m2mInfo.dev_children(maxDepth - 1), m2mInfo.dev_distances(maxDepth - 1), dev_M, p);





	return;

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < REPETITIONS; ++i) {
		P2MInfo p2mInfo{ numParticles / maxParticlesPerNode * 4 }; // Rough estimate for number of leaf nodes
		M2MInfo m2mInfo{ };
		auto timingMap = buildTree(particles.X(), numParticles, maxParticlesPerNode, domain, mortonCodes, p2mInfo, m2mInfo);
	}
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Tree building took: " << duration.count() / REPETITIONS << " ms" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < REPETITIONS; ++i) {
		std::vector<Interaction> p2pList;
		std::vector<Interaction> m2lList;
		dualTreeTraversal(treeMap, p2pList, m2lList);
	}
	end = std::chrono::high_resolution_clock::now();

	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Computing interactions took: " << duration.count() / REPETITIONS << " ms" << std::endl;

	std::vector<Interaction> p2pList;
	std::vector<Interaction> m2lList;
	dualTreeTraversal(treeMap, p2pList, m2lList);
	std::cout << "Total P2P interactions: " << p2pList.size() << std::endl;
	std::cout << "Total M2L interactions: " << m2lList.size() << std::endl;

	//bool write_volume_mesh(const std::string & path,
	//	const int dim,
	//	const int cell_size,
	//	const std::vector<double> &points,
	//	const std::vector<int> &elements);

    //particleBuffer.freeFields();

	//writeVTK(particles, numParticles, "particles.vtu", OutputType::X);
	
	//vpmio::writeOctreeVTK(treeMap, "octree_leaf_nodes.vtu");
}

static void sortByMorton(ParticleBuffer& particles, size_t numParticles, std::span<uint64_t> mortonCodes)
{
	// Create indices vector for sorting by morton indices
	std::vector<size_t> indices(numParticles);
	std::iota(indices.begin(), indices.end(), 0);

	// Sort indices by morton codes
	std::sort(std::execution::par_unseq, indices.begin(), indices.end(),
		[&mortonCodes](size_t a, size_t b) {
			return mortonCodes[a] < mortonCodes[b];
		});

	std::vector<bool> visited(indices.size(), false);

	for (size_t i = 0; i < indices.size(); ++i) {
		if (visited[i]) continue;
		if (indices[i] == i)
		{
			visited[i] = true;
			continue;
		}
		size_t j = i;
		while (!visited[j]) {
			visited[j] = true;
			size_t next = indices[j];
			if (!visited[next]) {
				std::swap(mortonCodes[j], mortonCodes[next]);
			}
			j = next;
		}
	}

	particles.permute(indices, BufferField::ALL);
}

static DomainInfo calcDomain(const vpm::vec3* points, size_t numPoints)
{
	vpm::vec3 lower = points[0];
	vpm::vec3 upper = points[0];
	for (size_t i = 1; i < numPoints; ++i) {
		lower = glm::min(lower, points[i]);
		upper = glm::max(upper, points[i]);
	}
	const vpm::real size = glm::compMax(upper - lower) * (1.0f + DOMAIN_EPSILON);
	return { lower, size };
}

static uint64_t spread_bits(uint64_t x) {
	x = (x | (x << 32)) & 0x001f00000000ffffULL;
	x = (x | (x << 16)) & 0x001f0000ff0000ffULL;
	x = (x | (x <<  8)) & 0x100f00f00f00f00fULL;
	x = (x | (x <<  4)) & 0x10c30c30c30c30c3ULL;
	x = (x | (x <<  2)) & 0x1249249249249249ULL;
	return x;
}

static std::vector<uint64_t> calcMortonCodes(const vpm::vec3* points, size_t numPoints, const DomainInfo& domain)
{
	std::vector<uint64_t> mortonCodes(numPoints);
	const float minNodeSize = domain.size / (1 << MAX_DEPTH);
	const uint64_t leadingBit = 1ULL << (3 * MAX_DEPTH);
	std::transform(std::execution::par_unseq,
		points, points + numPoints, mortonCodes.begin(),
		[&](const glm::fvec3& p) {
			uint64_t cx = static_cast<uint64_t>((p.x - domain.lower.x) / minNodeSize);
			uint64_t cy = static_cast<uint64_t>((p.y - domain.lower.y) / minNodeSize);
			uint64_t cz = static_cast<uint64_t>((p.z - domain.lower.z) / minNodeSize);
			return (spread_bits(cx) | (spread_bits(cy) << 1) | (spread_bits(cz) << 2))
				| leadingBit;
		});
	return mortonCodes;
}

struct TreeDebugInfo
{
	size_t count;
	size_t maxDepth;
};

std::unordered_map<uint64_t, HCell> buildTree(
	const vpm::vec3* points,
	size_t numPoints,
	int maxPointsPerNode,
	const DomainInfo& domain,
	const std::vector<uint64_t>& mortonCodes,
	P2MInfo& p2mInfo,
	M2MInfo& m2mInfo)
{
	std::array<glm::fvec3, MAX_DEPTH> nodeParentCenters;
	nodeParentCenters[0] = domain.lower + domain.size / 2.0f;
	std::array<vpm::real, MAX_DEPTH> nodeHalfSizes;
	std::array<vpm::real, MAX_DEPTH> nodeRadii;
	for (int i = 0; i < MAX_DEPTH; ++i) {
		nodeHalfSizes[i] = domain.size / (2.0f * (1 << i));
		nodeRadii[i] = nodeHalfSizes[i] * sqrt(3.0f);
	}
	static const vpm::vec3 childOffsets[8] = {
		vpm::vec3(-1.0f, -1.0f, -1.0f),
		vpm::vec3( 1.0f, -1.0f, -1.0f),
		vpm::vec3(-1.0f,  1.0f, -1.0f),
		vpm::vec3( 1.0f,  1.0f, -1.0f),
		vpm::vec3(-1.0f, -1.0f,  1.0f),
		vpm::vec3( 1.0f, -1.0f,  1.0f),
		vpm::vec3(-1.0f,  1.0f,  1.0f),
		vpm::vec3( 1.0f,  1.0f,  1.0f)
	};

	std::unordered_map<uint64_t, HCell> treeMap;
	treeMap.emplace(
		1, HCell{ 0, 1, nodeParentCenters[0], nodeRadii[0], 0, 0}
	);

	// Keep track of node numbers per depth
	std::array<size_t, MAX_DEPTH> depthCounts{};
	depthCounts[0] = 1; // Root node at depth 0	

	// Tracks the start index of points in the current node, initialized to 0 for the root node
	size_t startIdx = 0;
	// Tracks for each depth the end index of points in the current node
	std::array<size_t, MAX_DEPTH> nodeEndIndices;
	// Initialized with the total number of points for the root node
	nodeEndIndices[0] = numPoints;
	// Current depth and morton index, starting with first child of the root node
	uint8_t depth = 1;
	uint64_t mortonIndex = 1 << 3;

	TreeDebugInfo debugInfo{ 0, 0 };
	while (depth > 0)
	{
		debugInfo.maxDepth = std::max(debugInfo.maxDepth, static_cast<size_t>(depth));

		const size_t endIdx = std::lower_bound(
			mortonCodes.begin() + startIdx,
			mortonCodes.begin() + nodeEndIndices[depth - 1],
			(mortonIndex + 1) << (3 * (MAX_DEPTH - depth))
		) - mortonCodes.begin();

		const size_t numPointsInNode = endIdx - startIdx;

		nodeEndIndices[depth] = endIdx;

		const uint8_t childIndex = mortonIndex & CHILD_MASK;

		const vpm::vec3 nodeCenter = nodeParentCenters[depth - 1] + nodeHalfSizes[depth] * childOffsets[childIndex];
		const size_t nodeIndex = depthCounts[depth];

		if (numPointsInNode > 0)
		{
			treeMap[mortonIndex] = HCell{ nodeIndex, mortonIndex, nodeCenter, nodeRadii[depth], 0 , depth};
			depthCounts[depth]++;

			const size_t parentIndex = mortonIndex >> 3;

			// Enable child bit in parent node
			treeMap[parentIndex].childMask |= (1 << childIndex);

			m2mInfo.add(depth, parentIndex, nodeIndex, nodeParentCenters[depth - 1] - nodeCenter);
		}

		if (numPointsInNode > maxPointsPerNode) // This is a parent node
		{
			// Descend into the first child
			if (depth >= MAX_DEPTH - 1) {
				std::cout << "Warning: Maximum tree depth reached. Some nodes may contain more than " << maxPointsPerNode << " points." << std::endl;
				throw std::runtime_error("Exceeded maximum tree depth");
			}

			nodeParentCenters[depth] = nodeCenter;

			++depth;
			mortonIndex <<= 3;
		}
		else // This is a leaf node
		{
			if (numPointsInNode > 0)
			{
				//p2mInfo.add(nodeIndex, depth, startIdx, endIdx, nodeCenter);
				p2mInfo.add(nodeIndex, depth, endIdx, nodeCenter);

				debugInfo.count += numPointsInNode;
			}
			// Keep ascending if current node is last child of its parent
			while ((mortonIndex & CHILD_MASK) == CHILD_MASK)
			{
				--depth;
				mortonIndex >>= 3;
			}

			// Advance to the next node
			++mortonIndex;
			startIdx = nodeEndIndices[depth];
		}
	}

	//std::cout << "Tree Total cells in tree: " << treeMap.size() << std::endl;
	//std::cout << "Accounted for points: " << debugInfo.count << std::endl;
	//std::cout << "Maximum depth reached: " << debugInfo.maxDepth << std::endl;

	// Compute cumulative index offsets
	std::array<size_t, MAX_DEPTH> depthOffsets{};
	for (int i = 1; i < MAX_DEPTH; ++i) {
		depthOffsets[i] = depthOffsets[i - 1] + depthCounts[i - 1];
		std::cout << "Depth " << i << ": " << depthCounts[i] << " nodes, index offset " << depthOffsets[i] << std::endl;
	}

	// Update cell indices in tree structure
	for (auto& [morton, cell] : treeMap) {
		cell.index += depthOffsets[cell.depth];
	}

	// Update cell indices in interaction lists
	m2mInfo.addOffsets(depthOffsets);
	p2mInfo.addOffsets(depthOffsets);

	return treeMap;
}

static void getChildren(const HCell* cell, std::vector<const HCell*>& children,
						const std::unordered_map<uint64_t, HCell>& treeMap)
{
	const uint64_t baseIndex = cell->morton << 3;

	uint8_t mask = cell->childMask;
	while (mask) {
		int i = std::countr_zero(mask);
		children.push_back(&treeMap.at(baseIndex | i));
		mask &= mask - 1;
	}
}

static bool mac(const HCell* cellA, const HCell* cellB)
{
	const float distance = glm::length(cellA->center - cellB->center);

	const float radiusA = cellA->radius;
	const float radiusB = cellB->radius;

	if (radiusA > radiusB) {
		const float d = distance - radiusB;
		return (d > 0) && (radiusA < MAC_COEFFICIENT * d);
	}
	else {
		const float d = distance - radiusA;
		return (d > 0) && (radiusB < MAC_COEFFICIENT * d);
	}
}

typedef std::pair<const HCell*, const HCell*> CellPair;
void dualTreeTraversal(const std::unordered_map<uint64_t, HCell>& treeMap, std::vector<Interaction>& p2pList, std::vector<Interaction>& m2lList)
{
	std::vector<CellPair> stack;
	stack.reserve(36 * MAX_DEPTH); // Reserve space to avoid reallocations

	// Initialize with root node
	stack.push_back({ &treeMap.at(1), &treeMap.at(1)});

	std::vector<const HCell*> children;
	children.reserve(8); // Reserve space for 8 children to avoid reallocations

	while (!stack.empty())
	{
		const CellPair pair = stack.back();
		stack.pop_back();
		const HCell* cellA = pair.first;
		const HCell* cellB = pair.second;

		if (cellA->morton == cellB->morton)
		{
			if (cellA->childMask == 0)
			{
				p2pList.push_back({ cellA->index, cellB->index });
			}
			else
			{
				getChildren(cellA, children, treeMap);
				for (int i = 0; i < children.size(); ++i)
				{
					for (int j = i; j < children.size(); ++j)
					{
						stack.push_back({ children[i], children[j] });
					}
				}
				children.clear();
			}
			continue;
		}

		if (mac(cellA, cellB))
		{
			m2lList.push_back({ cellA->index, cellB->index });
			continue;
		}

		if ((cellA->childMask == 0) && (cellB->childMask == 0))
		{
			p2pList.push_back({ cellA->index, cellB->index });
			continue;
		}

		if ((cellA->childMask == 0) || ((cellB->childMask > 0) && cellA->radius <= cellB->radius))
		{
			getChildren(cellB, children, treeMap);
			for (const HCell* childB : children)
			{
				stack.push_back({ cellA, childB });
			}
			children.clear();
		}
		else
		{
			getChildren(cellA, children, treeMap);
			for (const HCell* childA : children)
			{
				stack.push_back({ childA, cellB });
			}
			children.clear();
		}
	}
}

