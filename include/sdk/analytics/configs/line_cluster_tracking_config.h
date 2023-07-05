/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_LINE_CLUSTER_TRACKING_CONFIG_H
#define METAVISION_SDK_ANALYTICS_LINE_CLUSTER_TRACKING_CONFIG_H

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Struct representing the parameters used to instantiate a LineClusterTracker inside
/// @ref PsmAlgorithm
struct LineClusterTrackingConfig {
    /// @brief Default constructor
    LineClusterTrackingConfig() = default;

    /// @brief Constructor
    /// @param precision_time_us Time duration between two asynchronous processes (us)
    /// @param bitsets_buffer_size Size of the bitset circular buffer
    /// (accumulation_time = bitsets_buffer_size *precision_time_us )
    /// @param cluster_ths Minimum width (in pixels) below which clusters of events are considered as noise
    /// @param num_clusters_ths Minimum number of cluster measurements below which a particle is considered as noise
    /// @param min_inter_clusters_distance Once small clusters have been removed, merge clusters that are closer than
    /// this distance. This helps dealing with dead pixels that could cut particles in half. If set to 0, do nothing
    /// @param learning_rate Ratio in the weighted mean between the current x position and the observation. This is used
    /// only when the particle is shrinking, because the front of the particle is always sharp while the trail might be
    /// noisy. 0.0 is conservative and does not take the observation into account, whereas 1.0 has no memory and
    /// overwrites the cluster estimate with the new observation. A value outside ]0,1] disables the weighted mean,
    /// and 1.0 is used instead.
    /// @param max_dx_allowed Caps x variation at this value. A negative value disables the clamping. This is used
    /// only when the particle is shrinking, because the front of the particle is always sharp while the trail might be
    /// noisy.
    /// @param max_nbr_empty_rows Number of consecutive empty measurements that is tolerated
    LineClusterTrackingConfig(unsigned int precision_time_us, unsigned int bitsets_buffer_size,
                              unsigned int cluster_ths = 3, unsigned int num_clusters_ths = 4,
                              unsigned int min_inter_clusters_distance = 1, float learning_rate = 1.f,
                              float max_dx_allowed = 5.f, unsigned int max_nbr_empty_rows = 0) :
        precision_time_us_(precision_time_us),
        bitsets_buffer_size_(bitsets_buffer_size),
        cluster_ths_(cluster_ths),
        num_clusters_ths_(num_clusters_ths),
        min_inter_clusters_distance_(min_inter_clusters_distance),
        learning_rate_(learning_rate),
        max_dx_allowed_(max_dx_allowed),
        max_nbr_empty_rows_(max_nbr_empty_rows) {}
    unsigned int precision_time_us_;
    unsigned int bitsets_buffer_size_;

    unsigned int cluster_ths_;
    unsigned int num_clusters_ths_;
    unsigned int min_inter_clusters_distance_;

    float learning_rate_;
    float max_dx_allowed_;
    unsigned int max_nbr_empty_rows_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_LINE_CLUSTER_TRACKING_CONFIG_H