#!/usr/bin/env sh

# BUILD_TYPE=RelWithDebInfo
BUILD_TYPE=Release
# BUILD_TYPE=Debug

cmake --build build/${BUILD_TYPE}
