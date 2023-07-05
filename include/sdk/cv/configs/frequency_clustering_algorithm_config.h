/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include "metavision/sdk/base/utils/timestamp.h"

#ifndef METAVISION_SDK_CV_FREQUENCY_CLUSTERING_ALGORITHM_CONFIG_H
#define METAVISION_SDK_CV_FREQUENCY_CLUSTERING_ALGORITHM_CONFIG_H

namespace Metavision {

/// @brief Data needed to configure a FrequencyClusteringAlgorithm
struct FrequencyClusteringAlgorithmConfig {
    /// @brief Constructor
    ///
    /// @param min_cluster_size Minimum size of a cluster to be output (in pixels)
    /// @param max_frequency_diff Maximum frequency difference for an input event to be
    /// associated to an existing cluster
    /// @param max_time_diff Maximum time difference to link an event to an existing cluster
    /// @param filter_alpha Filter weight for updating the cluster position with a new event
    FrequencyClusteringAlgorithmConfig(int min_cluster_size, float max_frequency_diff, timestamp max_time_diff,
                                       float filter_alpha) :
        min_cluster_size_(min_cluster_size),
        max_frequency_diff_(max_frequency_diff),
        max_time_diff_(max_time_diff),
        filter_alpha_(filter_alpha) {}

    /// @brief Default constructor
    FrequencyClusteringAlgorithmConfig() = default;

    int min_cluster_size_                = 1;
    float max_frequency_diff_            = 5;
    Metavision::timestamp max_time_diff_ = 1000;
    float filter_alpha_                  = 0.1f;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_FREQUENCY_CLUSTERING_ALGORITHM_CONFIG_H
