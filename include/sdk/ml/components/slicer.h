/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_SLICER_H
#define METAVISION_SDK_ML_SLICER_H

#include <functional>
#include <algorithm>
#include <vector>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief A slicer informs connected component of the slice end
///
/// The slicer receives as input:
///     - vector of events
///     - timestamp
///
/// When a vector of events is received the slicer does:
///     1. Call the preprocess function on all events
///     2. Look for end_slice_it (the last event of the current slice inside the vector)
///     3. If end_slice_it is not the vector end
///         4. Forward events in this slice to the event callbacks
///         5. Call the end of slice callbacks to signal the end of the slice
///         6. Return to second step
///     7. Forwards remaining events to the event callbacks
///
/// When a timestamp is received, the end of slice callbacks are
/// called as required to have a slice greater than this timestamp.
template<typename Event>
class Slicer {
public:
    /// Function type preprocessing event before calling the callbacks
    using PreProcessingEvent = std::function<void(const Event *, const Event *, std::vector<Event> &)>;

    /// Function type handling events
    using EventCallback = std::function<void(const Event *, const Event *)>;

    /// Function to indicate the end of the current slice
    using EndSliceCallback = std::function<void(timestamp)>;

    /// @brief Builds a slicer
    /// @param batch_time Duration of one slice
    /// @param preprocess Function to transform the events before calling event callbacks
    Slicer(timestamp batch_time, PreProcessingEvent preprocess = nullptr) :
        batch_time_(batch_time),
        next_batch_ts_(batch_time),
        preprocessing_(preprocess),
        event_callbacks_(0),
        end_slice_callbacks_(0),
        tmp_buffer_(256) {
        if (batch_time_ == 0) {
            throw std::runtime_error("Slicer cannot update under 1 microseconds."
                                     " batch_time_ must be greater than 0");
        }
    }

    /// @brief Adds function to be called on event reception and on batch end
    ///
    /// The callbacks will be executed all in the same thread thus
    /// the execution time of the callback should not be too long to
    /// keep it realtime
    ///
    /// @param new_event_callback Function called when events are available
    /// @param new_end_slice_callback Function called at each end of batch
    void add_callbacks(EventCallback new_event_callback, EndSliceCallback new_end_slice_callback) {
        if (new_event_callback) {
            event_callbacks_.push_back(new_event_callback);
        }
        if (new_end_slice_callback) {
            end_slice_callbacks_.push_back(new_end_slice_callback);
        }
    }

    /// @brief Returns function to be called on received events to ease lambda function creation
    ///
    /// The function returned on each call will loop until then end of events:
    ///    - call event callbacks on the events until the end of the batch
    ///    - call end batch callbacks if the batch end is reached
    ///
    /// @return Function of type EventCallback
    EventCallback get_event_callback() {
        return std::bind(&Slicer::event_callback, this, std::placeholders::_1, std::placeholders::_2);
    }

    /// @brief Returns function to be called time to time to update output
    /// @note Every event should be received for this timestamp
    /// @return Function of type EndSliceCallback
    /// @sa @ref timestamp_callback
    EndSliceCallback get_timestamp_callback() {
        return std::bind(&Slicer::timestamp_callback, this, std::placeholders::_1);
    }

    /// @brief Loops over all events to find slice ends and:
    ///    - call event callbacks on the events up to the end of the slices
    ///    - call end slice callbacks for each ended slice
    /// @param begin Pointer on the first event
    /// @param end Pointer on the last event
    void event_callback(Event const *begin, Event const *end) {
        Event const *preprocess_begin = begin;
        Event const *preprocess_end   = end;
        timestamp last_ts             = next_batch_ts_ - 1;
        if (begin != end) {
            last_ts = (end - 1)->t;
        }
        if (preprocessing_) {
            tmp_buffer_.clear();
            preprocessing_(preprocess_begin, preprocess_end, tmp_buffer_);
            preprocess_begin = tmp_buffer_.data();
            preprocess_end   = preprocess_begin + tmp_buffer_.size();
        }

        while (preprocess_begin < preprocess_end) {
            auto batch_end = split_batch(preprocess_begin, preprocess_end);
            if (batch_end != preprocess_begin) {
                for (auto &callback : event_callbacks_) {
                    callback(preprocess_begin, batch_end);
                }
            }
            if (batch_end != preprocess_end) {
                switch_batch();
            }
            preprocess_begin = batch_end;
        }

        timestamp_callback(last_ts);
    }

    /// @brief Calls all end batch callbacks for every ended batches
    /// @param time Current time reached to generate the required slices
    inline void timestamp_callback(timestamp time) {
        while (time >= next_batch_ts_) {
            switch_batch();
        }
    }

    /// @brief Sets timestamp of the first batch
    /// @param time Time at which the first slice begin
    void set_start_ts(timestamp time) {
        next_batch_ts_ = time + batch_time_;
    }

private:
    /// @brief Switches to the next slice by calling registered end_slices_callback.
    inline void switch_batch() {
        for (auto &end_callback : end_slice_callbacks_) {
            end_callback(next_batch_ts_);
        }
        next_batch_ts_ += batch_time_;
    }

    /// @brief Gets the last event belonging to the current batch
    /// @param begin First event of the buffer
    /// @param end Last event of the current buffer
    /// @return Last event belonging to the current batch
    inline Event const *split_batch(Event const *begin, Event const *end) {
        if (begin->t >= next_batch_ts_) {
            return begin;
        }
        return std::lower_bound(begin, end, next_batch_ts_,
                                [](const Event &ev, const timestamp &t) { return ev.t < t; });
    }

    const timestamp batch_time_;                        ///< duration of one batch
    timestamp next_batch_ts_;                           ///< last timestamp of the batch
    PreProcessingEvent preprocessing_;                  ///< function to be called before the callbacks
    std::vector<EventCallback> event_callbacks_;        ///< function to publish events
    std::vector<EndSliceCallback> end_slice_callbacks_; ///< function to notify the end of a batch
    std::vector<Event> tmp_buffer_;                     ///< temporary buffer used during one call
};

} // namespace Metavision

#endif // METAVISION_SDK_ML_SLICER_H
