# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

project(metavision_jet_monitoring_calibration)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS base driver cv core ui REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

set(sample metavision_jet_monitoring_calibration)
set(source_files ${sample}.cpp jet_monitoring_calibration_gui.h jet_monitoring_calibration_gui.cpp)
add_executable(${sample} ${source_files})
target_link_libraries(${sample}
    MetavisionSDK::base
    MetavisionSDK::driver
    MetavisionSDK::cv
    MetavisionSDK::core
    MetavisionSDK::ui
    Boost::program_options)
