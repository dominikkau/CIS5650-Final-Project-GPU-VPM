#pragma once

#include <cuda.h>
#include <vector>
#include <array>
#include "../vpmcore/common.h"

class P2MInfo
{
	std::vector<vpm::nidx_t> nodes_;
	std::vector<int> depths_;
	std::vector<vpm::pidx_t> pointsEnd_;
	std::vector<vpm::vec3> centers_;
	vpm::nidx_t* dev_nodes_ = nullptr;
	vpm::pidx_t* dev_pointsEnd_ = nullptr;
	vpm::vec3* dev_centers_ = nullptr;

public:
	P2MInfo(size_t capacity);
	~P2MInfo();

	size_t size() const { return nodes_.size(); }
	void add(vpm::nidx_t nodeIndex, int depth, vpm::pidx_t pointEndIndex, const vpm::vec3& nodeCenter);
	void addOffsets(const std::array<vpm::nidx_t, MAX_DEPTH>& depthOffsets);
	void toDevice() const;

	const std::vector<vpm::nidx_t>& nodes() const { return nodes_; }
	const std::vector<int>& depths() const { return depths_; }
	const std::vector<vpm::pidx_t>& pointsEnd() const { return pointsEnd_; }
	const std::vector<vpm::vec3>& centers() const { return centers_; }

	vpm::nidx_t* dev_nodes() const { return dev_nodes_; }
	vpm::pidx_t* dev_pointsEnd() const { return dev_pointsEnd_; }
	vpm::vec3* dev_centers() const { return dev_centers_; }
};

namespace fmm
{
	unsigned int shRequirementP2M(int p, int blockSize);
	__global__ void p2m(const vpm::nidx_t* __restrict__ nodes, const vpm::pidx_t* __restrict__ pointsEnd, const vpm::vec3* __restrict__ centers,
		size_t count, const vpm::vec3* __restrict__ xs, const vpm::real* __restrict__ qs, vpm::real* M, int p);
}