#pragma once

#include <glm/glm.hpp>

#define TRANSPOSED

__host__ __device__ inline glm::vec3 xDotNablaY(glm::vec3 x, glm::mat3 jacobianY)
{
#ifdef TRANSPOSED
    return jacobianY * vec3;
#else
    return vec3 * jacobianY;
#endif
}

__host__ __device__ inline glm::vec3 nablaCrossX(glm::mat3 jacobianX)
{
    return glm::vec{
        jacobianX[3, 2] - jacobianX[2, 3],
        jacobianX[1, 3] - jacobianX[3, 1],
        jacobianX[2, 1] - jacobianX[1, 2]
    }
}