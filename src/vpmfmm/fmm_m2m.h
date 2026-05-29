#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include "../vpmcore/common.h"

// M2M data for a single depth of the tree
struct M2MEntry
{
	std::vector<vpm::nidx_t> parents_;		// parent node indices
	std::vector<vpm::nidx_t> children_;		// child node indices
	std::vector<vpm::vec3> distances_;	// distance from child to parent (center of child - center of parent)

	vpm::nidx_t size() const { return parents_.size(); }
};

class M2MInfo
{
	std::vector<M2MEntry> entries_;
	vpm::nidx_t* dev_parents_ = nullptr;
	vpm::nidx_t* dev_children_ = nullptr;
	vpm::vec3* dev_distances_ = nullptr;

public:
	~M2MInfo();

	vpm::nidx_t size() const;
	vpm::nidx_t size(int depth) const { return entries_[depth].parents_.size(); }
	int depthCount() const { return entries_.size(); }
	void add(int depth, vpm::nidx_t parent, vpm::nidx_t child, const vpm::vec3& distance);
	void addOffsets(const std::array<vpm::nidx_t, MAX_DEPTH>& depthOffsets);
	void toDevice() const;

	const std::vector<vpm::nidx_t>& parents(int depth) const { return entries_[depth].parents_; }
	const std::vector<vpm::nidx_t>& children(int depth) const { return entries_[depth].children_; }
	const std::vector<vpm::vec3>& distances(int depth) const { return entries_[depth].distances_; }
	
	vpm::nidx_t* dev_parents(int depth) const;
	vpm::nidx_t* dev_children(int depth) const;
	vpm::vec3* dev_distances(int depth) const;
};

namespace fmm
{
	__global__ void m2m(const vpm::nidx_t* __restrict__ parents, const vpm::nidx_t* __restrict__ children, const vpm::vec3* __restrict__ distances, vpm::real* M, int p);
}
