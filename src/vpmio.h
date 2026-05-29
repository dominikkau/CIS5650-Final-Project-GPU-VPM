#pragma once

#include <string>
#include <unordered_map>
#include "vpmcore/particlebuffer.h"
#include "vpmfmm/tree.h"

namespace vpmio
{
    ParticleBuffer readParticleBuffer(const std::string& path);
    void writeParticleBuffer(const std::string& path, const ParticleBuffer& buffer, vpm::pidx_t numParticles, int fieldMask);
    void writeOctreeVTK(const std::unordered_map<vpm::midx_t, HCell>& treeMap,
        const std::string& outputPath);
}