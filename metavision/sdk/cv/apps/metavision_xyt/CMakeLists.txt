# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

project(metavision_xyt_demo)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

set (sample metavision_xyt)

find_package(MetavisionSDK COMPONENTS driver cv REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OGRE COMPONENTS Bites CONFIG REQUIRED)
find_package(imgui QUIET)

find_library(imgui_lib imgui)
find_library(freetype_lib freetype)

set(platform_specific_link)
set(imgui_interface_libs "freetype;stb")
if(UNIX)
    set(platform_specific_link X11)
elseif(WIN32)
    set(imgui_interface_libs "${freetype_lib}")
endif()

# When complied from source, Ogre3d comes with Imgui. But when installed from packages, 
# Imgui might not be set as dependencies. Because we rely on imgui, let's make sure we find it
# on the system
if(NOT imgui_FOUND) 
    find_file(imgui_header imgui.h PATH_SUFFIXES imgui)
    get_filename_component(imgui_header_dir ${imgui_header} DIRECTORY)

    add_library(imgui UNKNOWN IMPORTED GLOBAL) 
    set_target_properties(imgui PROPERTIES 
        IMPORTED_LOCATION "${imgui_lib}"
        IMPORTED_LINK_INTERFACE_LIBRARIES "${imgui_interface_libs}"
        INTERFACE_INCLUDE_DIRECTORIES "${imgui_header_dir}"
    )
    add_library(imgui::imgui ALIAS imgui)
endif()

add_executable(${sample} 
    ${sample}.cpp
    Polyfill/OgreImGuiInputListener.cpp
    Polyfill/OgreImGuiOverlay.cpp
)

target_link_libraries(${sample} OgreBites MetavisionSDK::cv MetavisionSDK::driver imgui::imgui Boost::program_options ${platform_specific_link})

if (WIN32)
    # Configure Ogre config files for a developper that is building the app from source
    get_filename_component(USER_DOCUMENTS "[HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders;personal]" ABSOLUTE)
    set(OGRE_INSTALL_DIR ${OGRE_PREFIX_DIR})
    if (CMAKE_BUILD_TYPE MATCHES ^[Dd][Ee][Bb][Uu][Gg]$)
        set(OGRE_INSTALL_DIR ${OGRE_INSTALL_DIR}/debug)
    endif()
    set(OGRE_INSTALL_PLUGIN_DIR ${OGRE_INSTALL_DIR}/bin)
    configure_file("resources.cfg.in" "${USER_DOCUMENTS}/Metavision-xyt/resources.cfg" @ONLY)
    configure_file("plugins.cfg.in" "${USER_DOCUMENTS}/Metavision-xyt/plugins.cfg" @ONLY)    
endif(WIN32)
