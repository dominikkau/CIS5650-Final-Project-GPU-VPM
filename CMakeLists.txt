cmake_minimum_required(VERSION 3.18)
project(GPU-VPM LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++11 for host code
# set(CMAKE_CXX_STANDARD 11)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(OpenGL REQUIRED)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

set(EIGEN_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(EIGEN REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

set(BOOST_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(BOOST REQUIRED)
include_directories(${BOOST_INCLUDE_DIRS})

set(headers
    src/cudaMat4.hpp
    src/glslUtility.hpp
    src/kernel.h
    src/main.hpp
    src/utilityCore.hpp
    src/FLOWVLM.h
    src/FLOWVLM_dt.h
    src/FLOWVLM_postprocessing.h
    src/FLOWVLM_rotor.h
    src/FLOWVLM_rotor_ccb.h
    src/FLOWVLM_solver.h
    src/FLOWVLM_tools.h
    src/FLOWVLM_wing.h
    src/FLOWVLM_wingsystem.h
    )

set(sources
    src/glslUtility.cpp
    src/kernel.cu
    src/main.cpp
    src/utilityCore.cpp
    src/FLOWVLM.cpp
    src/FLOWVLM_dt.cpp
    src/FLOWVLM_postprocessing.cpp
    src/FLOWVLM_rotor.cpp
    src/FLOWVLM_rotor_ccb.cpp
    src/FLOWVLM_solver.cpp
    src/FLOWVLM_tools.cpp
    src/FLOWVLM_wing.cpp
    src/FLOWVLM_wingsystem.cpp
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
target_link_libraries(${CMAKE_PROJECT_NAME} ${LIBRARIES})
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
endif()
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
    )
