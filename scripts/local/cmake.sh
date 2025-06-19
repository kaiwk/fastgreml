#!/usr/bin/env sh

BUILD_TYPE=RelWithDebInfo
# BUILD_TYPE=Release
# BUILD_TYPE=Debug

cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -B build/${BUILD_TYPE} -S .
