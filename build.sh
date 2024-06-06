#!/bin/bash

# Navigate to your source directory first if needed
# cd /path/to/your/source
mkdir 3rdparty
cd 3rdparty
wget -O Eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip Eigen.zip
cd eigen-3.4.0
mkdir build
cmake -DCMAKE_INSTALL_PREFIX=./install ..
cd ../../..
mkdir build
cd build
make clean
cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_CXX_STANDARD=20 -DCMAKE_INSTALL_PREFIX=./install -DEIGEN3_INCLUDE_DIR=../3rdparty/eigen-3.4.0/build/install -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON ..
make -j10
make install

