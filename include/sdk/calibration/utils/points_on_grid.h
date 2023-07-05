/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_POINTS_ON_GRID_H
#define METAVISION_SDK_CALIBRATION_POINTS_ON_GRID_H

#include <vector>
#include <opencv2/core/types.hpp>

namespace Metavision {

/// @brief Checks that points are on a grid.
/// @warning With strong radial distortion the threshold here should not be too aggressive.
/// Another possibility is to use are_points_on_grid_radial_distortion instead (recommended for fisheye)
/// @param centers The points of the grid
/// @param cols Number of horizontal lines
/// @param rows Number of vertical lines
/// @param ths Maximum line fitting error for a point to be considered on the grid. Default value is 0.01
/// @return true if points fit well the grid
bool are_points_on_grid(const std::vector<cv::Point2f> &centers, unsigned int cols, unsigned int rows,
                        float ths = 0.01f);

/// @brief Checks that points are on a grid (for fisheye)
/// @param centers The points of the grid
/// @param cols Number of horizontal lines
/// @param rows Number of vertical lines
/// @param cos_max_angle Cosine of maximum angle between 2 consecutive vectors (3 consecutive points) which belong
/// to the same row of the grid. Default value is 0.9 (~ 25 degrees)
/// @return true if points fit well the grid
bool are_points_on_grid_radial_distortion(const std::vector<cv::Point2f> &centers, unsigned int cols, unsigned int rows,
                                          float cos_max_angle = 0.9f);

} // namespace Metavision

#endif // METAVISION_SDK_CALIBRATION_POINTS_ON_GRID_H
