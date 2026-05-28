#include <fstream>
#include <sstream>
#include <iomanip>
#include "vpmfmm/tree.h"
#include "vpmio.h"
#include "lean_vtk.hpp"
#include "vpmcore/particlebuffer.h"

ParticleBuffer vpmio::readParticleBuffer(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + path);

    std::string line;

    // Read number of particles
    std::getline(file, line);
    size_t numParticles = std::stoull(line);

    // Read field mask
    std::getline(file, line);
    std::istringstream fieldStream(line);
    std::string fieldName;
    int fieldMask = 0;
    while (fieldStream >> fieldName)
    {
        if (fieldName == "X")     fieldMask |= BufferField::X;
        else if (fieldName == "Gamma") fieldMask |= BufferField::GAMMA;
        else if (fieldName == "sigma") fieldMask |= BufferField::SIGMA;
        else if (fieldName == "index") fieldMask |= BufferField::INDEX;
        else if (fieldName == "U")     fieldMask |= BufferField::U;
        else if (fieldName == "J")     fieldMask |= BufferField::J;
        else if (fieldName == "M")     fieldMask |= BufferField::M;
        else if (fieldName == "C")     fieldMask |= BufferField::C;
        else if (fieldName == "SFS")   fieldMask |= BufferField::SFS;
        else throw std::runtime_error("Unknown field: " + fieldName);
    }

    // Allocate buffer
    ParticleBuffer buffer{ ParticleBufferType::HOST, numParticles };
    buffer.mallocFields(fieldMask);

    // Helper lambdas
    auto readVec3 = [](const std::string& line, vpm::vec3& v) {
        std::istringstream ss(line);
        ss >> v.x >> v.y >> v.z;
        if (ss.fail()) throw std::runtime_error("Failed to parse vec3: " + line);
        };

    auto readMat3 = [](const std::string& line, vpm::mat3& m) {
        std::istringstream ss(line);
        // Column-major order to match GLM's layout
        ss >> m[0][0] >> m[0][1] >> m[0][2]
            >> m[1][0] >> m[1][1] >> m[1][2]
            >> m[2][0] >> m[2][1] >> m[2][2];
        if (ss.fail()) throw std::runtime_error("Failed to parse mat3: " + line);
        };

    // Read data for each field in order
    auto readField = [&](int field, auto readFn, auto* ptr) {
        if (!(fieldMask & field)) return;
        for (size_t i = 0; i < numParticles; ++i) {
            if (!std::getline(file, line))
                throw std::runtime_error("Unexpected end of file");
            readFn(line, ptr[i]);
        }
        };
    
    // Read fields in a consistent order
    readField(BufferField::X, readVec3, buffer.X());
    readField(BufferField::GAMMA, readVec3, buffer.Gamma());
    readField(BufferField::SIGMA, [](const std::string& line, vpm::real& v) {
        std::istringstream ss(line);
        ss >> v;
        if (ss.fail()) throw std::runtime_error("Failed to parse float: " + line);
        }, buffer.sigma());
    readField(BufferField::INDEX, [](const std::string& line, size_t& v) {
        std::istringstream ss(line);
        ss >> v;
        if (ss.fail()) throw std::runtime_error("Failed to parse int: " + line);
        }, buffer.index());
    readField(BufferField::U, readVec3, buffer.U());
    readField(BufferField::J, readMat3, buffer.J());
    readField(BufferField::M, readMat3, buffer.M());
    readField(BufferField::C, readVec3, buffer.C());
    readField(BufferField::SFS, readVec3, buffer.SFS());

    return buffer;
}

void vpmio::writeParticleBuffer(const std::string& path, const ParticleBuffer& buffer, size_t numParticles, int fieldMask)
{
    std::ofstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + path);

    // Write number of particles
    file << numParticles << "\n";

    // Write field names
    if (fieldMask & BufferField::X)     file << "X ";
    if (fieldMask & BufferField::GAMMA) file << "Gamma ";
    if (fieldMask & BufferField::SIGMA) file << "sigma ";
    if (fieldMask & BufferField::INDEX) file << "index ";
    if (fieldMask & BufferField::U)     file << "U ";
    if (fieldMask & BufferField::J)     file << "J ";
    if (fieldMask & BufferField::M)     file << "M ";
    if (fieldMask & BufferField::C)     file << "C ";
    if (fieldMask & BufferField::SFS)   file << "SFS ";
    file << "\n";

    file << std::setprecision(10);

    auto writeVec3 = [&](const vpm::vec3& v) {
        file << v.x << " " << v.y << " " << v.z << "\n";
        };

    auto writeMat3 = [&](const vpm::mat3& m) {
        file << m[0][0] << " " << m[0][1] << " " << m[0][2] << " "
            << m[1][0] << " " << m[1][1] << " " << m[1][2] << " "
            << m[2][0] << " " << m[2][1] << " " << m[2][2] << "\n";
        };

    // Write fields in same order as read
    if (fieldMask & BufferField::X)
        for (size_t i = 0; i < numParticles; ++i) writeVec3(buffer.X()[i]);
    if (fieldMask & BufferField::GAMMA)
        for (size_t i = 0; i < numParticles; ++i) writeVec3(buffer.Gamma()[i]);
    if (fieldMask & BufferField::SIGMA)
        for (size_t i = 0; i < numParticles; ++i) file << buffer.sigma()[i] << "\n";
    if (fieldMask & BufferField::INDEX)
        for (size_t i = 0; i < numParticles; ++i) file << buffer.index()[i] << "\n";
    if (fieldMask & BufferField::U)
        for (size_t i = 0; i < numParticles; ++i) writeVec3(buffer.U()[i]);
    if (fieldMask & BufferField::J)
        for (size_t i = 0; i < numParticles; ++i) writeMat3(buffer.J()[i]);
    if (fieldMask & BufferField::M)
        for (size_t i = 0; i < numParticles; ++i) writeMat3(buffer.M()[i]);
    if (fieldMask & BufferField::C)
        for (size_t i = 0; i < numParticles; ++i) writeVec3(buffer.C()[i]);
    if (fieldMask & BufferField::SFS)
        for (size_t i = 0; i < numParticles; ++i) writeVec3(buffer.SFS()[i]);
}

void vpmio::writeOctreeVTK(const std::unordered_map<uint64_t, HCell>& treeMap,
    const std::string& outputPath)
{
    leanvtk::VTUWriter writer;

    std::vector<double> points;
    std::vector<int> elements;
    std::vector<double> depthValues;
    std::vector<double> mortonIndices;

    int vertexIndex = 0;

    // Iterate through all nodes and extract leaf nodes
    for (const auto& [mortonIndex, cell] : treeMap) {
        if (!cell.childMask) {
            // Calculate depth from morton index
            int depth = 0;
            uint64_t temp = mortonIndex;
            while (temp > 1) {
                temp >>= 3;
                depth++;
            }

            // Calculate half-size of the box
            double halfSize = cell.radius / std::sqrt(3.0);  // radius is for sphere inscribed in cube

            // Calculate the 8 corners of the box centered at cell.center
            glm::dvec3 center(cell.center.x, cell.center.y, cell.center.z);
            glm::dvec3 corners[8] = {
                center + glm::dvec3(-halfSize, -halfSize, -halfSize),  // 0
                center + glm::dvec3(halfSize, -halfSize, -halfSize),  // 1
                center + glm::dvec3(halfSize,  halfSize, -halfSize),  // 2
                center + glm::dvec3(-halfSize,  halfSize, -halfSize),  // 3
                center + glm::dvec3(-halfSize, -halfSize,  halfSize),  // 4
                center + glm::dvec3(halfSize, -halfSize,  halfSize),  // 5
                center + glm::dvec3(halfSize,  halfSize,  halfSize),  // 6
                center + glm::dvec3(-halfSize,  halfSize,  halfSize)   // 7
            };

            // Add all 8 vertices to the points list
            for (int i = 0; i < 8; ++i) {
                points.push_back(corners[i].x);
                points.push_back(corners[i].y);
                points.push_back(corners[i].z);
            }

            // Add the element (hexahedron with 8 vertices)
            elements.push_back(vertexIndex + 0);
            elements.push_back(vertexIndex + 1);
            elements.push_back(vertexIndex + 2);
            elements.push_back(vertexIndex + 3);
            elements.push_back(vertexIndex + 4);
            elements.push_back(vertexIndex + 5);
            elements.push_back(vertexIndex + 6);
            elements.push_back(vertexIndex + 7);

            // Add depth value for this element
            depthValues.push_back(static_cast<double>(depth));

            mortonIndices.push_back(static_cast<double>(mortonIndex));

            vertexIndex += 8;
        }
    }

    // Write volume mesh
    if (!points.empty() && !elements.empty()) {
        std::ofstream outFile(outputPath);
        if (outFile.is_open()) {

            // Add depth as a cell (element) field
            writer.add_cell_scalar_field("Morton Index", mortonIndices);

            // Add depth as a cell (element) field
            writer.add_cell_scalar_field("Depth", depthValues);

            // cell_size = 8 for hexahedron (cube)
            writer.write_volume_mesh(outFile, 3, 8, points, elements);



            outFile.close();
            std::cout << "ok Octree leaf nodes as 3D mesh written to " << outputPath << std::endl;
            std::cout << "   Total leaf nodes (hexahedra): " << (vertexIndex / 8) << std::endl;
        }
        else {
            std::cerr << "FAIL Failed to open file: " << outputPath << std::endl;
        }
    }
    else {
        std::cout << "FAIL  No leaf nodes found in octree" << std::endl;
    }
}