#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include "../vpmcore/common.h"


namespace fmm
{
	__global__ void l2l(
		const vpm::nidx_t firstTarget,
		const vpm::nidx_t* __restrict__ sources,
		const vpm::vec3* __restrict__ distances,
		vpm::real* __restrict__ L,
		vpm::nidx_t count, int p);
}