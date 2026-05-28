#pragma once

#include <string>
#include "vpmcore/particlebuffer.h"

namespace vpmio
{
    ParticleBuffer readParticleBuffer(const std::string& path);
    void writeParticleBuffer(const std::string& path, const ParticleBuffer& buffer, size_t numParticles, int fieldMask);
    void writeOctreeVTK(const std::unordered_map<uint64_t, HCell>& treeMap,
        const std::string& outputPath);
}