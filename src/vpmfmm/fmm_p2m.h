#pragma once

#include <cuda.h>
#include <vector>
#include <array>
#include "../vpmcore/common.h"

class P2MInfo
{
	std::vector<size_t> nodes_;
	std::vector<int> depths_;
	std::vector<size_t> pointsEnd_;
	std::vector<vpm::vec3> centers_;
	size_t* dev_nodes_ = nullptr;
	size_t* dev_pointsEnd_ = nullptr;
	vpm::vec3* dev_centers_ = nullptr;

public:
	P2MInfo(size_t capacity);
	~P2MInfo();

	size_t size() const { return nodes_.size(); }
	void add(size_t nodeIndex, int depth, size_t pointEndIndex, const vpm::vec3& nodeCenter);
	void addOffsets(const std::array<size_t, MAX_DEPTH>& depthOffsets);
	void toDevice() const;

	const std::vector<size_t>& nodes() const { return nodes_; }
	const std::vector<int>& depths() const { return depths_; }
	const std::vector<size_t>& pointsEnd() const { return pointsEnd_; }
	const std::vector<vpm::vec3>& centers() const { return centers_; }

	size_t* dev_nodes() const { return dev_nodes_; }
	size_t* dev_pointsEnd() const { return dev_pointsEnd_; }
	vpm::vec3* dev_centers() const { return dev_centers_; }
};

namespace fmm
{
	__global__ void p2m(const size_t* nodes, const size_t* pointsEnd, const vpm::vec3* centers,
		size_t count, const vpm::vec3* xs, const vpm::vec3* qs, float* Rout, int p);
}