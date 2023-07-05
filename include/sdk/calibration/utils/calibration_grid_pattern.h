

/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_CALIBRATION_GRID_PATTERN_H
#define METAVISION_SDK_CALIBRATION_CALIBRATION_GRID_PATTERN_H

#include <vector>
#include <opencv2/core/types.hpp>

namespace Metavision {

/// @brief Structure representing the geometry of the rigid 3D pattern used for calibration.
struct CalibrationGridPattern {
    /// @brief Default constructor
    CalibrationGridPattern() = default;

    /// @brief Constructor using the physical size of cells
    /// @param n_cols Number of columns in the pattern (width)
    /// @param n_rows Number of rows of the pattern (height)
    /// @param dist_between_rows Distance between two consecutive rows of the pattern
    /// @param dist_between_cols Distance between two consecutive columns of the pattern
    CalibrationGridPattern(unsigned int n_cols, unsigned int n_rows, float dist_between_rows, float dist_between_cols);

    /// @brief Constructor using a flattened vector of 3D coordinates
    /// @param n_cols Number of columns in the pattern (width)
    /// @param n_rows Number of rows of the pattern (height)
    /// @param coordinates_3d 3D positions of each point of the pattern, should be n_cols * n_rows * 3 size
    CalibrationGridPattern(unsigned int n_cols, unsigned int n_rows, const std::vector<double> &coordinates_3d);

    /// @brief Constructor using a vector of 3D points
    /// @param n_cols Number of columns in the pattern (width)
    /// @param n_rows Number of rows of the pattern (height)
    /// @param pts_3d 3D points of the pattern, should be n_cols * n_rows size
    CalibrationGridPattern(unsigned int n_cols, unsigned int n_rows, const std::vector<cv::Point3f> &pts_3d);

    unsigned int n_cols_; ///< Pattern's width
    unsigned int n_rows_; ///< Pattern's height
    unsigned int n_pts_;  ///< Nbr of points in pattern

    float square_height_ = -1;                ///< Distance between two consecutive rows of the pattern
    float square_width_  = -1;                ///< Distance between two consecutive columns of the pattern
    std::vector<cv::Point3f> base_3D_points_; ///< 3D positions of each point of the pattern
};

} // namespace Metavision

#endif // METAVISION_SDK_CALIBRATION_CALIBRATION_GRID_PATTERN_H
