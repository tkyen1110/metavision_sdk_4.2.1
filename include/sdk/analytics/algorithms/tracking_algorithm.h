/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_TRACKING_ALGORITHM_H
#define METAVISION_SDK_ANALYTICS_TRACKING_ALGORITHM_H

#include <memory>
#include <vector>
#include <functional>
#include <sstream>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/analytics/events/event_tracking_data.h"

namespace Metavision {

class TrackingAlgorithmInternal;
struct TrackingConfig;

/// @brief Class that tracks objects using Metavision Tracking API
class TrackingAlgorithm {
public:
    using OutputBuffer = std::vector<EventTrackingData>;
    using OutputCb     = std::function<void(timestamp, OutputBuffer &)>;

    /// @brief Builds a new TrackingAlgorithm object.
    /// @param sensor_width Sensor's width.
    /// @param sensor_height Sensor's height.
    /// @param config Tracking's configuration.
    TrackingAlgorithm(int sensor_width, int sensor_height, TrackingConfig &config);

    /// @brief Default Destructor.
    ~TrackingAlgorithm();

    /// @brief Sets a callback to retrieve the list of tracked objects
    /// (see @ref EventTrackingData) when the tracker is updated
    /// (see @ref set_update_frequency).
    /// @note The generated vector will be passed to the callback as a non constant reference, meaning that the
    /// client is free to copy it or swap it. In case of a swap, the swapped vector will be automatically cleaned.
    /// @param output_cb Function to call
    void set_output_callback(const OutputCb &output_cb);

    /// @brief Gets the size of the smallest trackable object.
    std::uint16_t get_min_size() const;

    /// @brief Gets the size of the biggest trackable object.
    std::uint16_t get_max_size() const;

    /// @brief Gets the speed of the slowest trackable object.
    float get_min_speed() const;

    /// @brief Gets the speed of the fastest trackable object.
    float get_max_speed() const;

    /// @brief Sets the size of the smallest trackable object.
    /// @param min_size Size of the smallest object (in pixels)
    void set_min_size(std::uint16_t min_size);

    /// @brief Sets the size of the biggest trackable object.
    /// @param max_size Size of the biggest object (in pixels)
    void set_max_size(std::uint16_t max_size);

    /// @brief Sets the speed of the slowest trackable object.
    /// @param min_speed Speed of the slowest object
    void set_min_speed(float min_speed);

    /// @brief Sets the speed of the fastest trackable object.
    /// @param max_speed Speed of the fastest object
    void set_max_speed(float max_speed);

    /// @brief Sets the frequency at which the algorithm generates the output.
    /// @param freq Frequency to generate the output
    void set_update_frequency(float freq);

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end);

private:
    std::unique_ptr<TrackingAlgorithmInternal> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_TRACKING_ALGORITHM_H
