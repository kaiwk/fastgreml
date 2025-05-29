cmake_minimum_required(VERSION 3.16)

project(fastgreml VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS  OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

# Fetch 3rd parties
include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Eigen)

# Find OpenMP
if(APPLE)
  # Use manual flags for OpenMP with Apple Clang
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_C_LIB_NAMES "omp")
  set(OpenMP_C_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_CXX_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
  set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

find_package(OpenMP REQUIRED)

# Source
file(GLOB_RECURSE SOURCES src/*.cpp src/*.cc src/*.h src/*.hpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)

if(APPLE)
  if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
    # For Eigen backend
    find_library(ACCELERATE_LIBRARY Accelerate)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ACCELERATE_LIBRARY})
  elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    # Do nothing
  endif()
endif()

target_link_libraries(${PROJECT_NAME}
  PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX
)

