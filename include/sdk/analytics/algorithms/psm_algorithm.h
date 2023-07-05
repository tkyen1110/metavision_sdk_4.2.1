/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_PSM_ALGORITHM_H
#define METAVISION_SDK_ANALYTICS_PSM_ALGORITHM_H

#include <functional>
#include <map>
#include <memory>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/analytics/configs/line_particle_tracking_config.h"
#include "metavision/sdk/analytics/configs/line_cluster_tracking_config.h"
#include "metavision/sdk/analytics/utils/line_cluster.h"
#include "metavision/sdk/analytics/utils/line_particle_tracking_output.h"

namespace Metavision {

class PsmAlgorithmInternal;

/// @brief Class that both counts objects and estimates their size using Metavision Particle Size Measurement API
///
class PsmAlgorithm {
public:
    using LineClustersOutput = std::vector<LineClusterWithId>;

    /// Type for callbacks called after each asynchronous process
    using OutputCb = std::function<void(const timestamp, LineParticleTrackingOutput &, LineClustersOutput &)>;

    /// @brief Constructor
    /// @param sensor_width Sensor's width (in pixels)
    /// @param sensor_height Sensor's height (in pixels)
    /// @param rows Rows on which to instantiate line cluster trackers
    /// @param detection_config Detection config
    /// @param tracking_config Tracking config
    /// @param num_process_before_matching Accumulate detected particles during n process before actually matching them
    /// to existing tracks
    PsmAlgorithm(int sensor_width, int sensor_height, const std::vector<int> &rows,
                 const LineClusterTrackingConfig &detection_config, const LineParticleTrackingConfig &tracking_config,
                 int num_process_before_matching);

    ~PsmAlgorithm();

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param ts End timestamp of the buffer. Used if higher than the timestamp of the last event
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename InputIt>
    void process_events(const timestamp ts, InputIt it_begin, InputIt it_end);

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end);

    /// @brief Function to pass a callback to get updated instances of @ref LineParticleTrackingOutput for particles
    /// count, sizes and trajectories, and @ref LineCluster for the event-clusters aggregated along the rows
    /// @param output_cb Function to call
    /// @note The generated objects will be passed to the callback as a non constant references, meaning that the
    /// user is free to copy it or swap it using std::swap. In case of a swap with a non initialized object, it will be
    /// automatically initialized
    void set_output_callback(const OutputCb &output_cb);

    /// @brief Resets line cluster trackers and line particle trackers
    void reset();

private:
    std::unique_ptr<PsmAlgorithmInternal> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_PSM_ALGORITHM_H
