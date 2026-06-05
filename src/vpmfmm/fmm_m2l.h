#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include "../vpmcore/common.h"

class M2LInfo
{
	std::vector<vpm::nidx_t> targets_;
	std::vector<vpm::nidx_t> sources_;
	std::vector<vpm::vec3> distances_;
	vpm::nidx_t* dev_targets_ = nullptr;
	vpm::nidx_t* dev_sources_ = nullptr;
	vpm::vec3* dev_distances_ = nullptr;

public:
	~M2LInfo();

	vpm::nidx_t size() const { return targets_.size(); };
	void add(vpm::nidx_t target, vpm::nidx_t source, const vpm::vec3& distance);
	void toDevice() const;

	const std::vector<vpm::nidx_t>& targets() const { return targets_; }
	const std::vector<vpm::nidx_t>& sources() const { return sources_; }
	const std::vector<vpm::vec3>& distances() const { return distances_; }

	vpm::nidx_t* dev_targets() const { return dev_targets_; };
	vpm::nidx_t* dev_sources() const { return dev_sources_; };
	vpm::vec3* dev_distances() const { return dev_distances_; };
};

namespace fmm
{
	__global__ void m2l(
		const vpm::nidx_t* __restrict__ targets,
		const vpm::nidx_t* __restrict__ sources,
		const vpm::vec3* __restrict__ distances,
		const vpm::real* __restrict__ M,
		vpm::real* __restrict__ L,
		vpm::nidx_t count, int p, bool reversed
	);
}
