cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(timemachine LANGUAGES CXX CUDA)

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

string(APPEND CMAKE_CUDA_FLAGS "-Xptxas -v -arch=sm_60 -O3 -lineinfo")
# string(APPEND CMAKE_CUDA_FLAGS "-Xptxas -v -arch=sm_60 -g -G -lineinfo")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
	set(CMAKE_INSTALL_PREFIX ${PARENT_DIR} CACHE PATH "Default Install Path" FORCE)
endif()

set(LIBRARY_NAME custom_ops)

# clone pybind if we can't find it in the build dir
set(PYBIND_SRC_DIR pybind11)

if(NOT EXISTS ${PYBIND_SRC_DIR})
  execute_process(COMMAND git clone --branch tm0.1 https://github.com/proteneer/pybind11.git ${PYBIND_SRC_DIR})
endif()

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/${PYBIND_SRC_DIR})

# NO_EXTRAS is needed since cuda doesn't use flto
pybind11_add_module(${LIBRARY_NAME} SHARED NO_EXTRAS
  src/wrap_kernels.cpp
  src/gradient.cu
  src/gpu_utils.cu
  src/nonbonded.cu
  src/neighborlist.cu
  src/restraint.cu
  src/harmonic_bond.cu
  src/harmonic_angle.cu
  src/periodic_torsion.cu
  src/integrator.cu
  src/stepper.cu
  src/context.cu
  src/gbsa.cu
  src/kernels/k_find_block_bounds.cu
)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(src/kernels)

set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_STANDARD 11)
target_link_libraries(${LIBRARY_NAME} -lcurand -lcudart)
set_target_properties(${LIBRARY_NAME} PROPERTIES PREFIX "")

install(TARGETS ${LIBRARY_NAME} DESTINATION "lib")

