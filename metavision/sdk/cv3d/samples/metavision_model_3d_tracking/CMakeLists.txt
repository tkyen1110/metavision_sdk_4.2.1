# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

project(metavision_model_3d_tracking)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS cv3d driver ui REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

set (sample metavision_model_3d_tracking)
set(source_files ${sample}.cpp)

add_executable(${sample} ${source_files})

target_include_directories(${sample} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${sample}
    MetavisionSDK::cv3d
    MetavisionSDK::driver
    MetavisionSDK::ui
    Boost::program_options)
