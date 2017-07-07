#!/bin/sh

# OgmaNeo
# Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
#
# This copy of OgmaNeo is licensed to you under the terms described
# in the OGMANEO_LICENSE.md file included in this distribution.

set -ex

#----------------------------------------
# Install the latest SWIG
cd $TRAVIS_BUILD_DIR

if [ $TRAVIS_OS_NAME == 'osx' ]; then
    brew upgrade pcre
    brew install swig
else
    sudo apt install libpcre3 libpcre3-dev
    sudo apt install swig3.0
fi
