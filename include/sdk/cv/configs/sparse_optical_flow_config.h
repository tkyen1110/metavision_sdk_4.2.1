/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_SPARSE_OPTICAL_FLOW_CONFIG_H
#define METAVISION_SDK_CV_SPARSE_OPTICAL_FLOW_CONFIG_H

#include "metavision/sdk/base/events/event_cd.h"

namespace Metavision {

/// @brief Configuration used when using the sparse optical flow algorithm
struct SparseOpticalFlowConfig {
    enum class Preset { SlowObjects, FastObjects };

    SparseOpticalFlowConfig(Preset preset) {
        if (preset == Preset::FastObjects) {
            /**
             *@brief distance_gain gain of the low pass filter to compute the size of a CCL
             */
            distance_gain = 0.05f;
            /**
             * @brief damping parameter of the Luenberger estimator, will determine of fast / slow the speed of a CCL
             * will converge
             */
            damping = 0.707f;
            /**
             * @brief omega_cutoff parameter of the Luenberger estimator, will determine how fast /slow the speed of a
             * CCL will converge
             */
            omega_cutoff = 10.f;
            /**
             * @brief min_cluster_size minimal size of a cluster in terms of number of events that hit it before being
             * outputted
             */
            int min_cluster_size = 10;
            /**
             * @brief max_link_time maximum time in us for two events to be linked in time
             */
            max_link_time = 20000;

            /**
             * @brief match_polarity should we have multi-polarity cluster or mono polarity ones
             */
            match_polarity = true;
            /**
             * @brief use_simple_match generally put this to true, otherwise will use a costly constant velocity match
             * strategy
             */
            use_simple_match = true;
            /**
             * @brief full_square should we check connectivity on the full 3x3 square around the events or just a cross
             * around it
             */
            full_square = false;
            /**
             * @brief last_event_only will only check the connectivity with the last event at every pixel positions
             */
            last_event_only = false;
            /**
             * @brief size_threshold threshold on the spatial size of a cluster before being outputted. Juts put it to a
             * big value
             */
            size_threshold = 100000000;
        } else if (preset == Preset::SlowObjects) {
            distance_gain      = 0.05f;
            damping            = 0.707f;
            omega_cutoff       = 5.f;
            min_cluster_size   = 5;
            max_link_time      = 50000;
            match_polarity     = true;
            use_simple_match   = true;
            full_square        = true;
            last_event_only    = false;
            int size_threshold = 100000000;
        }
    }

    /// @brief Constructor
    /// @param distance_gain distance gain of the low pass filter to compute the size of a CCL
    /// @param damping damping parameter of the Luenberger estimator, will determine of fast / slow the speed of a CCL
    /// will converge
    /// @param omega_cutoff omega_cutoff parameter of the Luenberger estimator, will determine how fast /slow the speed
    /// of a CCL will converge
    /// @param min_cluster_size min_cluster_size minimal size of a cluster in terms of number of events that hit it
    /// before being outputted
    /// @param max_link_time max_link_time maximum time in us for two events to be linked in time
    /// @param match_polarity should we have multi-polarity cluster or mono polarity ones
    /// @param use_simple_match generally put this to true, otherwise will use a costly constant velocity match strategy
    /// @param full_square generally put this to true, otherwise will use a costly constant velocity match strategy
    /// @param last_event_only last_event_only will only check the connectivity with the last event at every pixel
    /// positions
    /// @param size_threshold size_threshold threshold on the spatial size of a cluster before being outputted. Juts put
    /// it to a big value
    SparseOpticalFlowConfig(float distance_gain = 0.05f, float damping = 0.707f, float omega_cutoff = 5.f,
                            unsigned int min_cluster_size = 5, timestamp max_link_time = 50000,
                            bool match_polarity = true, bool use_simple_match = true, bool full_square = true,
                            bool last_event_only = false, unsigned int size_threshold = 100000000) :
        distance_gain(distance_gain),
        damping(damping),
        omega_cutoff(omega_cutoff),
        min_cluster_size(min_cluster_size),
        max_link_time(max_link_time),
        match_polarity(match_polarity),
        use_simple_match(use_simple_match),
        full_square(full_square),
        last_event_only(last_event_only),
        size_threshold(size_threshold) {}

    float distance_gain;
    float damping;
    float omega_cutoff;
    unsigned int min_cluster_size;
    timestamp max_link_time;
    bool match_polarity;
    bool use_simple_match;
    bool full_square;
    bool last_event_only;
    unsigned int size_threshold;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_SPARSE_OPTICAL_FLOW_CONFIG_H
