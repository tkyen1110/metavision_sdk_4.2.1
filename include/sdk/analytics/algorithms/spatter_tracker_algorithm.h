/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_ALGORITHM_H
#define METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_ALGORITHM_H

#include <opencv2/opencv.hpp>
#include <functional>
#include <memory>
#include <vector>

#include "metavision/sdk/analytics/configs/spatter_tracker_algorithm_config.h"
#include "metavision/sdk/analytics/events/event_spatter_cluster.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/events/event_cd.h"

namespace Metavision {

class SpatterTrackerAlgorithmInternal;

/// @brief Class that tracks spatter clusters using Metavision SpatterTracking API
class SpatterTrackerAlgorithm {
public:
    /// Type of the callback to access the output cluster events
    using TrackerCb = std::function<void(const timestamp, const std::vector<EventSpatterCluster> &)>;

    /// @brief Builds a new @ref SpatterTrackerAlgorithm object
    /// @param width Sensor's width (in pixels)
    /// @param height Sensor's height (in pixels)
    /// @param config Spatter tracker's configuration
    SpatterTrackerAlgorithm(int width, int height, const SpatterTrackerAlgorithmConfig &config);

    /// @brief Default destructor
    ~SpatterTrackerAlgorithm();

    /// @brief Registers a callback to get the output cluster events
    /// @param tracker_cb Function to call
    void set_output_callback(const TrackerCb &tracker_cb);

    /// @brief Sets the region that isn't processed
    /// @param center Center of the region
    /// @param radius Radius of the region
    void set_nozone(cv::Point center, int radius);

    /// @brief Returns the current number of clusters
    /// @return The current number of clusters
    int get_cluster_count() const;

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end);

private:
    std::unique_ptr<SpatterTrackerAlgorithmInternal> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_ALGORITHM_H
