/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef ROI_LINES_EVENTS_LOADER_H
#define ROI_LINES_EVENTS_LOADER_H

#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_set>

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/core/algorithms/polarity_filter_algorithm.h>
#include <metavision/sdk/cv/algorithms/activity_noise_filter_algorithm.h>
#include <metavision/sdk/cv/algorithms/transpose_events_algorithm.h>

namespace Metavision {

/// @brief Class that opens a RAW file, applies filtering algorithms on it and buffers all the events belonging to a
/// given set of lines of interest
///
/// In addition to defining lines of interest, here are the optional filtering steps:
/// - use a polarity filter
/// - use an activity filter
/// - transpose events
/// - crop events inside a time interval
class RoiLineEventsLoader {
public:
    /// @brief Constructor
    /// @param rows Lines of interest
    RoiLineEventsLoader(const std::vector<int> &rows);

    /// @brief Loads events from Input RAW file using only the specified lines of interest
    /// @param input_raw_path Path to the input RAW file
    /// @param output_events Output events belonging to the lines of interest
    /// @param output_camera_size Output camera size
    /// @param transpose_axis Set to true to rotate the camera 90 degrees clockwise in case of particles moving
    /// horizontally in FOV
    /// @param polarity Process only events of this polarity
    /// @param activity_time_ths Length of the time window for activity filtering (in us)
    /// @param process_from Start time to process events (in us)
    /// @param process_to End time to process events (in us)
    bool load_events_from_rawfile(const std::string &input_raw_path, std::vector<EventCD> &output_events,
                                  cv::Size &output_camera_size, bool transpose_axis = false, short polarity = 0,
                                  timestamp activity_time_ths = 0, timestamp process_from = 0,
                                  timestamp process_to = -1);

private:
    /// @brief Sets camera and filtering algorithms
    /// @param input_raw_path Path to the input RAW file
    /// @param transpose_axis Set to true to rotate the camera 90 degrees clockwise in case of particles moving
    /// horizontally in FOV
    /// @param polarity Process only events of this polarity
    /// @param activity_time_ths Length of the time window for activity filtering (in us)
    /// @param output_camera_size Output camera size
    /// @return false if the initialization failed
    bool init_camera_and_filters(const std::string &input_raw_path, bool transpose_axis, short polarity,
                                 timestamp activity_time_ths, cv::Size &output_camera_size);

    /// @brief Processes the events coming from the camera
    /// @param begin Iterator to the first event to process
    /// @param end Iterator to the past-end event to process
    /// @param output_events Container used to buffer the filtered camera events
    void camera_callback(const EventCD *begin, const EventCD *end, std::vector<EventCD> &output_events);

    /// @brief Stores the filtered camera events if they belong to the lines of interest
    /// @param it_begin Iterator to the first event to store
    /// @param it_end Iterator to the past-end event to store
    /// @param output_events Container used to buffer the filtered camera events
    template<typename InputIt>
    void store_events(InputIt it_begin, InputIt it_end, std::vector<EventCD> &output_events);

    Metavision::Camera camera_;    ///< Camera
    std::unordered_set<int> rows_; ///< Lines of interest

    // Time interval
    timestamp process_from_ = 0;  ///< Start time to process events (in us)
    timestamp process_to_   = -1; ///< End time to process events (in us)

    // Conditional variables to notify the end of the processing
    std::condition_variable process_cond_;
    std::mutex process_mutex_;
    volatile bool is_processing_ = true;

    // Filtering
    std::vector<EventCD> filtered_buffer_;
    std::unique_ptr<PolarityFilterAlgorithm> polarity_filter_;              ///< Filter by polarity
    std::unique_ptr<TransposeEventsAlgorithm> transpose_events_filter_;     ///< Transpose X/Y on events
    std::unique_ptr<ActivityNoiseFilterAlgorithm<>> activity_noise_filter_; ///< Filter noisy events
};

template<typename InputIt>
void RoiLineEventsLoader::store_events(InputIt it_begin, InputIt it_end, std::vector<EventCD> &output_events) {
    for (auto it = it_begin; it != it_end; ++it) {
        // Check if it belongs to a line of interest
        if (rows_.find(it->y) != rows_.end())
            output_events.emplace_back(*it);
    }
}

} // namespace Metavision

#endif // ROI_LINES_EVENTS_LOADER_H
