/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <metavision/sdk/base/utils/log.h>

#include "roi_line_events_loader.h"

namespace Metavision {

RoiLineEventsLoader::RoiLineEventsLoader(const std::vector<int> &rows) {
    // Fill set
    for (int r : rows)
        rows_.insert(r);
}

bool RoiLineEventsLoader::load_events_from_rawfile(const std::string &input_raw_path,
                                                   std::vector<EventCD> &output_events, cv::Size &output_camera_size,
                                                   bool transpose_axis, short polarity, timestamp activity_time_ths,
                                                   timestamp process_from, timestamp process_to) {
    std::unique_lock<std::mutex> lock(process_mutex_); // Prevent multiple calls

    is_processing_ = true;
    output_events.clear();
    process_from_ = process_from;
    process_to_   = process_to;

    if (!init_camera_and_filters(input_raw_path, transpose_axis, polarity, activity_time_ths, output_camera_size))
        return false;

    // Add camera callbacks
    camera_.add_runtime_error_callback([](const Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); });

    camera_.cd().add_callback([this, &output_events](const EventCD *begin, const EventCD *end) {
        camera_callback(begin, end, output_events);
    });

    // Stop the pipeline when the camera is stopped
    camera_.add_status_change_callback([this](const Metavision::CameraStatus &status) {
        if (status == Metavision::CameraStatus::STOPPED && is_processing_) {
            std::lock_guard<std::mutex> lock(process_mutex_);
            is_processing_ = false;
            process_cond_.notify_all();
        }
    });

    if (!camera_.start()) {
        MV_LOG_ERROR() << "The camera could not be started.";
        return false;
    }

    // Wait until the end of the file
    process_cond_.wait(lock, [this] { return !is_processing_; });

    try {
        camera_.stop();
    } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }

    return true;
}

bool RoiLineEventsLoader::init_camera_and_filters(const std::string &input_raw_path, bool transpose_axis,
                                                  short polarity, timestamp activity_time_ths,
                                                  cv::Size &output_camera_size) {
    // Create camera
    try {
        camera_ = Metavision::Camera::from_file(
            input_raw_path,
            Metavision::FileConfigHints().real_time_playback(false)); // Run as fast as possible
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return false;
    }

    // Initialize preprocessing filters
    const auto &geometry = camera_.geometry();

    if (transpose_axis) {
        transpose_events_filter_  = std::make_unique<TransposeEventsAlgorithm>();
        output_camera_size.width  = geometry.height();
        output_camera_size.height = geometry.width();
    } else {
        output_camera_size.width  = geometry.width();
        output_camera_size.height = geometry.height();
    }

    if (activity_time_ths != 0) {
        activity_noise_filter_ = std::make_unique<ActivityNoiseFilterAlgorithm<>>(
            output_camera_size.width, output_camera_size.height, activity_time_ths);
    }

    if (polarity >= 0)
        polarity_filter_ = std::make_unique<PolarityFilterAlgorithm>(polarity);

    return true;
}

/// Function that applies a filter on events if it's not a nullptr.
/// The boolean parameter is_first indicates whether the filter should be applied on the raw events
/// or on the events that have already been filtered through other filters.
template<class AlgoPtr, class InputIt, class FilteredIt>
inline void apply_filter_if_enabled(const AlgoPtr &algo, InputIt &begin, InputIt &end,
                                    std::vector<FilteredIt> &output_buffer, bool &is_first) {
    if (algo) {
        if (is_first) {
            assert(begin != nullptr && end != nullptr);
            output_buffer.clear();
            algo->process_events(begin, end, std::back_inserter(output_buffer));
            is_first = false; // the next filters will have to use output_buffer instead of begin and end
            begin    = nullptr;
            end      = nullptr;
        } else {
            auto end_it = algo->process_events(output_buffer.cbegin(), output_buffer.cend(), output_buffer.begin());
            output_buffer.resize(std::distance(output_buffer.begin(), end_it));
        }
    }
}

void RoiLineEventsLoader::camera_callback(const EventCD *begin, const EventCD *end,
                                          std::vector<EventCD> &output_events) {
    if (!is_processing_)
        return;

    // Adjust iterators to make sure we only process a given range of timestamps [process_from_, process_to_]
    // Get iterator to the first element greater or equal than process_from_
    begin = std::lower_bound(begin, end, process_from_,
                             [](const Metavision::EventCD &ev, timestamp ts) { return ev.t < ts; });

    // Get iterator to the first element greater than process_to_
    if (process_to_ >= 0) {
        if (begin != end && std::prev(end)->t > process_to_) {
            std::lock_guard<std::mutex> lock(process_mutex_);
            is_processing_ = false;
            process_cond_.notify_all();
            return;
        }

        end = std::lower_bound(begin, end, process_to_,
                               [](const Metavision::EventCD &ev, timestamp ts) { return ev.t <= ts; });
    }
    if (begin == end)
        return;

    // Apply filters
    bool is_first = true;
    apply_filter_if_enabled(polarity_filter_, begin, end, filtered_buffer_, is_first);
    apply_filter_if_enabled(transpose_events_filter_, begin, end, filtered_buffer_, is_first);
    apply_filter_if_enabled(activity_noise_filter_, begin, end, filtered_buffer_, is_first);

    // Process filtered events
    if (is_first)
        store_events(begin, end, output_events);
    else
        store_events(filtered_buffer_.cbegin(), filtered_buffer_.cend(), output_events);
}

} // namespace Metavision
