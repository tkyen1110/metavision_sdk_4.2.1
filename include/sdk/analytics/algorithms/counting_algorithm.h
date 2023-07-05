/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_COUNTING_ALGORITHM_H
#define METAVISION_SDK_ANALYTICS_COUNTING_ALGORITHM_H

#include <functional>
#include <memory>
#include <opencv2/core/types.hpp>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/events/event_cd.h"

namespace Metavision {

class CountingAlgorithmInternal;
class LineCounter;
struct MonoCountingStatus;

/// @brief Class to count objects using Metavision Counting API
class CountingAlgorithm {
public:
    using OutputEvent = std::pair<timestamp, MonoCountingStatus>;
    /// Type of the callback to access the last count
    using OutputCb = std::function<void(const OutputEvent &)>;

    /// @brief Constructor
    /// @param width Sensor's width (in pixels)
    /// @param height Sensor's height (in pixels)
    /// @param cluster_ths Minimum width (in pixels) below which clusters of events are considered as noise
    /// @param accumulation_time_us Accumulation time of the event buffer before processing (in us)
    CountingAlgorithm(int width, int height, int cluster_ths, Metavision::timestamp accumulation_time_us = 1);

    /// @brief Default destructor
    ~CountingAlgorithm();

    /// @brief Adds a new line to count objects
    /// @param row Specifies which row (y coordinate in pixels) is used to count
    void add_line_counter(int row);

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end);

    /// @brief Function to pass a callback to get the last count
    /// @param output_cb Function to call
    void set_output_callback(const OutputCb &output_cb);

    /// @brief Resets the count of all lines
    void reset_counters();

private:
    std::unique_ptr<CountingAlgorithmInternal> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_COUNTING_ALGORITHM_H
