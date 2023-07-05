/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_ALGORITHM_CONFIG_H
#define METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_ALGORITHM_CONFIG_H

#include <limits>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Configuration to instantiate a SpatterTrackerAlgorithmConfig
struct SpatterTrackerAlgorithmConfig {
    /// @brief Constructor
    /// @param cell_width Width of cells used for clustering
    /// @param cell_height Height of cells used for clustering
    /// @param accumulation_time Time to accumulate events and process
    /// @param untracked_threshold Maximum number of times a spatter_cluster can stay untracked before being removed
    /// @param activation_threshold Threshold distinguishing an active cell from inactive cells (i.e. minimum number of
    /// events in a cell to consider it as active)
    /// @param apply_filter If true, than applying a simple filter to remove crazy pixels
    /// @param max_distance Max distance for clusters association (in pixels)
    /// @param min_size Minimum object size (in pixels) - minimum of the object's width and height should be larger than
    /// this value
    /// @param max_size Maximum object size (in pixels) - maximum of the object's width and height should be smaller
    /// than this value
    SpatterTrackerAlgorithmConfig(int cell_width, int cell_height, timestamp accumulation_time,
                                  int untracked_threshold = 5, int activation_threshold = 10, bool apply_filter = true,
                                  int max_distance = 50, const int min_size = 1,
                                  const int max_size = std::numeric_limits<int>::max()) :
        cell_width_(cell_width),
        cell_height_(cell_height),
        accumulation_time_(accumulation_time),
        untracked_threshold_(untracked_threshold),
        activation_threshold_(activation_threshold),
        apply_filter_(apply_filter),
        max_distance_(max_distance),
        min_size_(min_size),
        max_size_(max_size) {}

    int cell_width_;
    int cell_height_;
    timestamp accumulation_time_;
    int untracked_threshold_;
    int activation_threshold_;
    bool apply_filter_;
    int max_distance_;
    int min_size_;
    int max_size_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_ALGORITHM_CONFIG_H
