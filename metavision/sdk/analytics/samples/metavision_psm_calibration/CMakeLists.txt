# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

project(metavision_psm_calibration)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS base driver cv core analytics REQUIRED)
find_package(Boost COMPONENTS filesystem program_options REQUIRED)
find_package(OpenCV COMPONENTS core highgui imgproc REQUIRED)

set(sample metavision_psm_calibration)

set(sources ${sources} ${sample}.cpp
                       roi_line_events_loader.cpp
                       psm_config_csv_loader.cpp
                       psm_display_simulator.cpp
                       time_slice_cursor.cpp
                       )

add_executable(${sample} ${sources})

target_link_libraries(${sample}
    MetavisionSDK::base
    MetavisionSDK::driver
    MetavisionSDK::cv
    MetavisionSDK::core
    MetavisionSDK::analytics
    Boost::program_options)
