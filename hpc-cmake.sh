#!/usr/bin/env sh


module load cmake eigen intelmkl

cmake -DBUILD_ON_HPC=ON \
    -DCMAKE_PREFIX_PATH="/soft/devtools/eigen-3.4.0;${CMAKE_PREFIX_PATH}" \
    -DMKL_DIR="/soft/compiler/intel/oneapi-2022.2/mkl/2022.1.0/lib/cmake/mkl" \
    -DCMAKE_BUILD_TYPE=Release -B build/Release -S .

