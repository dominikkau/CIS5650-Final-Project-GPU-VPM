#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#define TRANSPOSED

// Constants
// const float PI = 3.14159265358979f;
// const float const1 = pow(1.0f / (2.0f * PI), 1.5f);
// const float const2 = sqrt(2.0f / PI);
// float const3 = 3.0f / (4.0f * PI);
// const float const4 = 1.0f / (4.0f * PI);
// const float sqrt2 = sqrt(2.0f);

#define PI 3.14159265358979f
#define const1 0.06349363593424097f
#define const2 0.7978845608028654f
#define const3 0.238732414637843f
#define const4 0.07957747154594767f
#define sqrt2 1.4142135623730951f

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