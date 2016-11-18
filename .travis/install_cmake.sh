#!/bin/sh

# OgmaNeo
# Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
#
# This copy of OgmaNeo is licensed to you under the terms described
# in the OGMANEO_LICENSE.md file included in this distribution.

set -ex

#----------------------------------------
# Install the latest CMake
cd $TRAVIS_BUILD_DIR

# Check to see if CMake cache folder is empty
if [ ! -d "$HOME/.local/cmake" ]; then
    wget https://cmake.org/files/v3.6/cmake-3.6.2.tar.gz
    tar -xzf cmake-3.6.2.tar.gz
    cd cmake-3.6.2

    if [ $TRAVIS_OS_NAME == 'osx' ]; then
        CC=clang CXX=clang++ ./configure --prefix=$HOME/.local/cmake
    else
        CC=gcc-4.8 CXX=g++-4.8 ./configure --prefix=$HOME/.local/cmake
    fi

    make
    make install
else
    echo "Using cached CMake directory."
fi
