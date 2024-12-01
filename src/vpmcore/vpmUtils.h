#pragma once

#include <glm/glm.hpp>

#define TRANSPOSED

__host__ __device__ inline glm::vec3 xDotNablaY(glm::vec3 x, glm::mat3 jacobianY) {
#ifdef TRANSPOSED
    return jacobianY * x;
#else
    return x * jacobianY;
#endif
}

__host__ __device__ inline glm::vec3 nablaCrossX(glm::mat3 jacobianX) {
    return glm::vec3{
        jacobianX[1][2] - jacobianX[2][1],
        jacobianX[2][0] - jacobianX[0][2],
        jacobianX[0][1] - jacobianX[1][0]
    };
}