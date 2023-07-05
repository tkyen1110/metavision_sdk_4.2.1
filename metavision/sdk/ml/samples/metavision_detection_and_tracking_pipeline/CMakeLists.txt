# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.


project(metavision_detection_and_tracking_pipeline)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

find_package(MetavisionSDK COMPONENTS base ml REQUIRED)
find_package(Boost COMPONENTS filesystem program_options REQUIRED)

set (sample metavision_detection_and_tracking_pipeline)
add_executable(${sample} ${sample}.cpp)

target_link_libraries(${sample}
    PRIVATE
    MetavisionSDK::base
    MetavisionSDK::ml
    Boost::filesystem
    Boost::program_options
    "${TORCH_LIBRARIES}"
    )

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET metavision_detection_and_tracking_pipeline
					 POST_BUILD
					 COMMAND ${CMAKE_COMMAND} -E copy_if_different
					 ${TORCH_DLLS}
					 $<TARGET_FILE_DIR:metavision_detection_and_tracking_pipeline>)
endif (MSVC)

