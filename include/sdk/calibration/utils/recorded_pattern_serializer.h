/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_RECORDED_PATTERN_SERIALIZER_H
#define METAVISION_SDK_RECORDED_PATTERN_SERIALIZER_H

#include "metavision/sdk/calibration/utils/calibration_grid_pattern.h"

namespace Metavision {

/// @brief Reads patterns from a JSON file located in the given directory
/// @param json_path Where to look for the JSON file
/// @param img_size [output] Image size
/// @param pts_2d [output] 2d pattern detections
/// @param pattern_3d [output] 3d pattern geometry
/// @return true if everything went according to the plan, false otherwise
bool read_patterns_from_file(const std::string &json_path, cv::Size &img_size,
                             std::vector<std::vector<cv::Point2f>> &pts_2d, CalibrationGridPattern &pattern_3d);

/// @brief Write patterns in a JSON file located in the given directory.
/// @param json_path Where the file will be written. The existence of this directory should preferably be ensured before
/// the call of this function
/// @param img_size Image size to save
/// @param pts_2d 2d pattern detections to save
/// @param pattern_3d 3d pattern geometry to save
/// @return true if everything went according to the plan, false otherwise.
bool write_patterns_to_file(const std::string &json_path, const cv::Size &img_size,
                            const std::vector<std::vector<cv::Point2f>> &pts_2d,
                            const CalibrationGridPattern &pattern_3d);

} // namespace Metavision

#endif
