#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "../vpmcore/common.h"

constexpr int MAX_P = 20; // Maximum expansion order, adjust as needed

namespace fmm
{
	// Number of swap coefficients needed for p-order expansions
	constexpr int countSwapCoefs(int p)
	{
		return (p * (p + 1) * (2 * p + 1)) / 6 - p * (p - 1) / 2;
	}

	__constant__ vpm::real c_swapCoefsM[countSwapCoefs(MAX_P)];
	__constant__ vpm::real c_swapCoefsL[countSwapCoefs(MAX_P)];

	void writeSwapCoefs(int p);

	// Rotate p-order local or multipole expansion M around z-axis by alpha = atan2(b, a) with c = sqrt(a^2 + b^2)
	__device__ __forceinline__ void rotateZ(vpm::real* M, vpm::real a, vpm::real b, vpm::real c, int p)
	{
		vpm::real tmp;
		vpm::real c1 = a / c;
		vpm::real s1 = b / c;
		vpm::real cm = 1.0f;
		vpm::real sm = 0.0f;
		for (int m = 1; m < p; ++m)
		{
			tmp = cm * c1 + sm * s1;
			sm = sm * c1 - cm * s1;
			cm = tmp;
			for (int n = m; n < p; ++n)
			{
				const int ip = (n * (n + 1)) + m;
				const int im = (n * (n + 1)) - m;

				tmp = M[ip] * cm - M[im] * sm;
				M[im] = M[im] * cm + M[ip] * sm;
				M[ip] = tmp;
			}
		}
	}

	template<const vpm::real* swapCoefs>
	__device__ __forceinline__ void swapXZ(vpm::real* M, vpm::real* r, int p)
	{
		int swapIdx = 0;

		for (int n = 0; n < p; ++n)
		{
			const int idx = (n * (n + 1));
			const bool oddN = n & 1;

			// Cache row from s_M to s_r
			for (int m = -n; m <= n; ++m)
			{
				r[n + m] = M[idx + m];
			}

			// Positive swap coefficients
			for (int m = 0; m <= n; ++m)
			{
				vpm::real tmp = 0.0f;
				for (int l = (oddN != (m & 1)); l <= n; l += 2)
				{
					tmp += r[n + l] * swapCoefs[swapIdx++];
				}
				M[idx + m] = tmp;
			}

			// Negative swap coefficients
			for (int m = 1; m <= n; ++m)
			{
				vpm::real tmp = 0.0f;
				for (int l = (oddN != (m & 1)); l <= n; l += 2)
				{
					tmp += r[n - l] * swapCoefs[swapIdx++];
				}
				M[idx - m] = tmp;
			}
		}
	}
}