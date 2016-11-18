#!/bin/sh

# OgmaNeo
# Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
#
# This copy of OgmaNeo is licensed to you under the terms described
# in the OGMANEO_LICENSE.md file included in this distribution.

set -ex

#----------------------------------------
# Package updates

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    #brew unlink cmake
    brew update

    # OSX has:
    # apple-gcc42
    brew tap homebrew/versions
    brew install gcc48

else
    sudo apt-get -qq update

    # Trusty has:
    # gcc 4.8.4
    # llvm clang 3.5.0
fi
