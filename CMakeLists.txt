cmake_minimum_required(VERSION 3.16)

project(fastgreml VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS  OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

# HPC is Linux environment
option(BUILD_ON_HPC OFF)

if(UNIX AND NOT APPLE)
  set(LINUX TRUE)
endif()

# Fetch 3rd parties
include(FetchContent)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(FETCHCONTENT_QUIET OFF)

if(NOT BUILD_ON_HPC)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(Eigen)
endif()

# Fetch spdlog
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.3)

FetchContent_MakeAvailable(spdlog)

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
  find_package(OpenMP REQUIRED)
endif()

if(LINUX AND NOT BUILD_ON_HPC)
  set(OMP_LIBRARY "/usr/lib/libgomp.so")
  set(MKL_DIR "/opt/intel/mkl/lib/intel64/cmake/mkl")
  set(MKL_THREADING "gnu_thread")
endif()

find_package(OpenMP REQUIRED)
find_package(MKL REQUIRED)
find_package(Eigen3 REQUIRED)

FetchContent_Declare(args
  GIT_REPOSITORY https://github.com/Taywee/args.git
  GIT_TAG        6.4.7
  GIT_PROGRESS   TRUE
  GIT_SHALLOW    TRUE
  EXCLUDE_FROM_ALL
  FIND_PACKAGE_ARGS NAMES args
)
FetchContent_GetProperties(args)
FetchContent_MakeAvailable(args)

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

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen taywee::args)

if(LINUX)
  target_compile_options(${PROJECT_NAME} PRIVATE -DMKL_ILP64)
  target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp)
  target_compile_options(${PROJECT_NAME} PRIVATE -march=native)
  target_link_libraries(${PROJECT_NAME} PRIVATE MKL::MKL)
  # Keep frame pointer for Flamegraph
  # target_compile_options(${PROJECT_NAME} PRIVATE -fno-omit-frame-pointer)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
  # Suppress warning
  target_compile_options(${PROJECT_NAME} PRIVATE -fpermissive)
endif()

if(APPLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog)
