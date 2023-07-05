/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_BLINKING_DOTS_GRID_DETECTOR_ALGORITHM_CONFIG_H
#define METAVISION_SDK_CALIBRATION_BLINKING_DOTS_GRID_DETECTOR_ALGORITHM_CONFIG_H

#include "metavision/sdk/base/utils/timestamp.h"
namespace Metavision {

/// @brief Structure containing the configuration to initialize a @ref BlinkingDotsGridDetectorAlgorithm.
struct BlinkingDotsGridDetectorAlgorithmConfig {
    ///
    /// @brief Determines the output periodicity of the algorithm. The callback will be called every
    /// @ref processing_timestep microseconds. If set to 0, the execution will be synchronous,
    /// meaning that the callback will always be called once at the end of each call to @ref
    /// BlinkingDotsGridDetectorAlgorithm::process_events.
    timestamp processing_timestep = 0;

    /// @brief Number of horizontal lines of points in the grid.
    int num_rows = 0;

    /// @brief Number of vertical lines of points in the grid.
    int num_cols = 0;

    /// @brief Distance between two consecutive rows in the grid
    float distance_between_rows = 1.f;

    /// @brief Distance between two consecutive columns in the grid
    float distance_between_cols = 1.f;

    /// @brief Special frequency, frequency of the first row of the grid, in Hz.
    float special_freq = 0.f;

    /// @brief Normal frequency, in Hz.
    float normal_freq = 0.f;

    /// @brief For the frequency estimation, the maximum difference between two successive periods to
    /// be considered the same (in us).
    timestamp period_diff_thresh_us = 2000;

    /// @brief Minimum number of successive stable periods to validate a frequency.
    int frequency_filter_length = 7;

    /// @brief Filter constant for the position of the center of the cluster. This value must be > 0 and <=1.
    /// Values closer to 1 produce a more reactive, but noisier position estimation.
    float cluster_center_filter_alpha = 0.05f;

    /// @brief Maximum difference to add a frequency event to a cluster, in Hz.
    float max_cluster_frequency_diff = 10.f;

    /// @brief Minimum size of a frequency cluster to be used (in pixels). For a LED grid, larger values can
    /// help filtering small clusters produced by reflections, but might prevent detection when the leds are
    /// far away.
    int min_cluster_size = 20;

    /// @brief Allows the detection of strongly distorted grids, as when using a fisheye lens.
    bool fisheye = false;
};

} // namespace Metavision

#endif // METAVISION_SDK_CALIBRATION_BLINKING_DOTS_GRID_DETECTOR_ALGORITHM_CONFIG_H
