#pragma once
#include "FLOWVLM_wing.h"

class FLOWVLM {
public:
	void solve(Wing& wing,
		std::function<std::vector<double>(const std::vector<double>&, double)> Vinf,
		double t = 0.0,
		std::function<Eigen::Vector3d(const std::vector<double>&, double)> vortexsheet = nullptr,
		std::function<std::vector<double>(int, double)> extraVinf = nullptr,
		bool keep_sol = false,
		const std::vector<double> & extraVinfArgs = {});
	std::vector<Horseshoe> getHorseshoes(Wing& wing, double t, std::function<std::vector<double>(int, double)> extraVinf, const std::vector<double>& extraVinfArgs);
	std::vector<double> Vind(Wing& wing, const std::vector<double>& X, double t, bool ign_col, bool ign_infvortex, bool only_infvortex);
	std::string get_hash(const std::string& var);
private:

};