#ifndef WING_H
#define WING_H

#include <glm/glm.hpp>
#include <vector>

// Function to create vertices for a simple wing geometry
std::vector<glm::vec3> createWingVertices(float chord, float span);

#endif // WING_H
