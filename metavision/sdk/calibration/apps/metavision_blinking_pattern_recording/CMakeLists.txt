# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

project(metavision_blinking_pattern_recording)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS core driver cv calibration ui REQUIRED)
find_package(Boost COMPONENTS filesystem program_options REQUIRED)

set (sample metavision_blinking_pattern_recording)
add_executable(${sample} ${sample}.cpp)
target_include_directories(${sample} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${sample}
    MetavisionSDK::core
    MetavisionSDK::driver
    MetavisionSDK::cv
    MetavisionSDK::calibration
    MetavisionSDK::ui
    Boost::program_options
    Boost::filesystem)
