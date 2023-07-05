/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_CAMERA_GEOMETRY_FACTORY_H
#define METAVISION_SDK_CV_CAMERA_GEOMETRY_FACTORY_H

#include <memory>
#include <string>

namespace Metavision {

template<typename T>
class CameraGeometryBase;

/// @brief Loads a camera geometry from a JSON file
/// @tparam T Either float or double
/// @param json_path Path to the JSON file containing the camera geometry
/// @return The camera geometry in case of success or nullptr otherwise
template<typename T>
std::unique_ptr<CameraGeometryBase<T>> load_camera_geometry(const std::string &json_path);

} // namespace Metavision

#endif // METAVISION_SDK_CV_CAMERA_GEOMETRY_FACTORY_H
