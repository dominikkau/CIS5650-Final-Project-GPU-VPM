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
	void toDevice();

	const auto& nodes() const { return nodes_; }
	const auto& depths() const { return depths_; }
	const auto& pointsEnd() const { return pointsEnd_; }
	const auto& centers() const { return centers_; }

	const auto nodes(vpm::nidx_t i) const { return nodes_[i]; }
	const auto depths(vpm::nidx_t i) const { return depths_[i]; }
	const auto pointsEnd(vpm::nidx_t i) const { return pointsEnd_[i]; }
	const auto centers(vpm::nidx_t i) const { return centers_[i]; }

	auto* dev_nodes() const { return dev_nodes_; }
	auto* dev_pointsEnd() const { return dev_pointsEnd_; }
	auto* dev_centers() const { return dev_centers_; }
};

namespace fmm
{
	unsigned int shRequirementP2M(int p, int blockSize);

	__global__ void p2m(
		const vpm::nidx_t* __restrict__ nodes,
		const vpm::pidx_t* __restrict__ pointsEnd,
		const vpm::vec3* __restrict__ centers,
		size_t count,
		const vpm::vec3* __restrict__ xs,
		const vpm::real* __restrict__ qs,
		vpm::real* __restrict__ M,
		int p);

	__global__ void l2p(
		const vpm::nidx_t* __restrict__ nodes,
		const vpm::pidx_t* __restrict__ pointsEnd,
		const vpm::vec3* __restrict__ centers,
		vpm::nidx_t count,
		const vpm::vec3* __restrict__ xs,
		vpm::real* __restrict__ ys,
		const vpm::real* __restrict__ L,
		int p);
}