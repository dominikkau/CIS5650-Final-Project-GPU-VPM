#include "wing.h"

// Implement the function that generates vertices for the wing
std::vector<glm::vec3> createWingVertices(float chord, float span) {
    std::vector<glm::vec3> vertices = {
        glm::vec3(0.0f, 0.0f, 0.0f),           // Leading edge left
        glm::vec3(0.0f, span, 0.0f),           // Leading edge right
        glm::vec3(chord, span, 0.0f),          // Trailing edge right
        glm::vec3(chord, 0.0f, 0.0f)           // Trailing edge left
    };
    return vertices;
}
