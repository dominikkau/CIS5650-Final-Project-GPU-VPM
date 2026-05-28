cmake_minimum_required(VERSION 3.18)
project(GPU-VPM LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++20 for host code
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

set(headers
    src/main.hpp
    src/vpmio.h
    src/vpmcore/common.h
    src/vpmcore/vpmmain.h
    src/vpmcore/particlebuffer.h
    src/vpmcore/particlefield.h
    src/vpmcore/kernels.h
    src/vpmcore/relaxation.h
    src/vpmcore/sfs.h
	src/vpmfmm/fmm_p2m.h
    src/vpmfmm/fmm_m2m.h
	src/vpmfmm/fmm_utils.h
	src/vpmfmm/tree.h
    src/lean_vtk.hpp
    src/vortexringsimulation.hpp
    src/roundjetsimulation.hpp
    )

set(sources
    src/main.cpp
    src/vpmio.cu
    src/vpmcore/vpmmain.cu
    src/vpmcore/particlebuffer.cu
    src/vpmcore/particlefield.cu
    src/vpmcore/relaxation.cu
    src/vpmcore/sfs.cu
	src/vpmfmm/fmm_p2m.cu
    src/vpmfmm/fmm_m2m.cu
    src/vpmfmm/fmm_utils.cu
	src/vpmfmm/tree.cu
    src/lean_vtk.cpp
    src/vortexringsimulation.cpp
    src/roundjetsimulation.cpp
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
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 75)
endif()
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})