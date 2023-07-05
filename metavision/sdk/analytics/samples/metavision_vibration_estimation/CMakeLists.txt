# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

project(metavision_vibration_estimation)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS base driver cv core analytics ui REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

set (sample metavision_vibration_estimation)
add_executable(${sample} ${sample}.cpp)
target_link_libraries(${sample}
    MetavisionSDK::base
    MetavisionSDK::driver
    MetavisionSDK::cv
    MetavisionSDK::core
    MetavisionSDK::analytics
    MetavisionSDK::ui
    Boost::program_options)