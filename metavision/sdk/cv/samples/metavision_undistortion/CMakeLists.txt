# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

project(metavision_undistortion)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS cv REQUIRED)
find_package(OpenCV COMPONENTS core imgproc imgcodecs REQUIRED)

set(FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(constants.cpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/constants.cpp @ONLY)

set (sample metavision_undistortion)
add_executable(${sample} ${sample}.cpp constants.h ${CMAKE_CURRENT_BINARY_DIR}/constants.cpp)
target_include_directories(${sample} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${sample} MetavisionSDK::cv opencv_core opencv_imgproc opencv_imgcodecs)
