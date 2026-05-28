#include <vector>
#include <stdexcept>
#include <iostream>
#include "fmm_utils.h"

// Low-effort tensor class for conveniently storing swap coefficients during recursion
struct Tensor3D {
	int d0, d1, d2;
	std::vector<vpm::real> data;

	Tensor3D(int d0, int d1, int d2, vpm::real init = 0.0)
		: d0(d0), d1(d1), d2(d2), data(d0* d1* d2, init) {
	}

	vpm::real& operator()(int i, int j, int k) {
		if (j < 0) j += d1;
		if (k < 0) k += d2;
		return data[i * d1 * d2 + j * d2 + k];
	}
	const vpm::real& operator()(int i, int j, int k) const {
		if (j < 0) j += d1;
		if (k < 0) k += d2;
		return data[i * d1 * d2 + j * d2 + k];
	}
};

Tensor3D calcSwapCoefs(int p, bool transpose)
{
	Tensor3D t(p, 2 * p - 1, 2 * p - 1);

	// Recursive calculation of swap coefficients for complex expansion coefficients
	t(0, 0, 0) = 1.0f;
	for (int n = 0; n < p - 1; ++n)
	{
		for (int l = -n-1; l <= n+1; ++l)
		{
			for (int m = -n; m <= n; ++m)
			{
				t(n + 1, m,     l) = 0.5f * (t(n, m, l - 1)                  - t(n, m, l + 1));
				t(n + 1, m + 1, l) = 0.5f * (t(n, m, l - 1) + 2 * t(n, m, l) + t(n, m, l + 1));
				t(n + 1, m - 1, l) = 0.5f * (t(n, m, l - 1) - 2 * t(n, m, l) + t(n, m, l + 1));
			}
		}
	}

	// Transpose coefficients to get swap coefficients for multipole expansion coefficients
	if (transpose)
	{
		for (int n = 0; n < p; ++n)
		{
			for (int m = -n; m <= n; ++m)
			{
				for (int l = -n; l < m; ++l)
				{
					std::swap(t(n, m, l), t(n, l, m));
				}
			}
		}
	}

	// Calculate swap coefficients for real expansion coefficients
	for (int n = 0; n < p; ++n)
	{
		int m1tol = 1;
		for (int l = 1; l < n + 1; ++l)
		{
			m1tol = -m1tol;

			t(n, 0, l) = t(n, 0, l) + m1tol * t(n, 0, -l);

			t(n, 0, -l) = 0.0f;

			int m1tom = 1;
			for (int m = 1; m < n + 1; ++m)
			{
				m1tom = -m1tom;
				t(n, -m, 0) = 0.0f;

				t(n, m,  l) = t(n, m, l) + m1tol * t(n, m, -l);
				t(n, m, -l) = 0.0f;

				t(n, -m, -l) = m1tom * (m1tol * t(n, -m, -l) - t(n, -m, l));
				t(n, -m,  l) = 0.0f;
			}
		}
	}

	return t;
}

void fmm::writeSwapCoefs(int p)
{
	if (p > MAX_P) throw std::runtime_error("p exceeds MAX_P");

	Tensor3D tM = calcSwapCoefs(p, true);
	Tensor3D tL = calcSwapCoefs(p, false);

	std::vector<vpm::real> swapCoefsM;
	std::vector<vpm::real> swapCoefsL;

	for (int n = 0; n < p; ++n)
	{
		const bool oddN = n & 1;
		for (int m = 0; m <= n; ++m)
		{
			for (int l = (oddN != (m & 1)); l <= n; l += 2)
			{
				swapCoefsM.push_back(tM(n, m, l));
				swapCoefsL.push_back(tL(n, m, l));
			}
		}
		for (int m = 1; m <= n; ++m)
		{
			for (int l = 1 + (oddN != (m & 1)); l <= n; l += 2)
			{
				swapCoefsM.push_back(tM(n, -m, -l));
				swapCoefsL.push_back(tL(n, -m, -l));
			}
		}
	}

	cudaMemcpyToSymbol(c_swapCoefsM, swapCoefsM.data(), swapCoefsM.size() * sizeof(vpm::real));
	cudaMemcpyToSymbol(c_swapCoefsL, swapCoefsL.data(), swapCoefsL.size() * sizeof(vpm::real));
}