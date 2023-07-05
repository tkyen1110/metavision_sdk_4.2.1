/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <algorithm>
#include <assert.h>
#include "time_slice_cursor.h"

namespace Metavision {

TimeSliceCursor::TimeSliceCursor(std::vector<EventCD> &&event_buffer, int precision_time, int buffer_duration,
                                 int num_process_before_matching) :
    event_buffer_(std::move(event_buffer)),
    min_events_ts_(event_buffer_.front().t),
    max_events_ts_(event_buffer_.back().t) {
    assert(std::is_sorted(event_buffer_.cbegin(), event_buffer_.cend(),
                          [](const auto &l, const auto &r) { return l.t < r.t; }));

    update_parameters(precision_time, buffer_duration, num_process_before_matching);

    // Initialize the timeslice at the beginning
    crt_ts_ = -1;
    go_to_begin();
}

void TimeSliceCursor::update_parameters(int precision_time, int buffer_duration, int num_process_before_matching) {
    assert(0 < precision_time);
    assert(0 < buffer_duration);
    assert(0 < num_process_before_matching);
    precision_time_              = precision_time;
    buffer_duration_             = buffer_duration;
    num_process_before_matching_ = num_process_before_matching;

    // Extreme timestamps (Make sure last particles are correctly detected)
    min_allowed_ts_ = precision_time * (1 + min_events_ts_ / precision_time);
    max_allowed_ts_ =
        precision_time * (max_events_ts_ / precision_time + buffer_duration + (num_process_before_matching + 1));

    // Set current ts to the previous multiple of precision time and clamp it between extreme values
    go_to(crt_ts_);
}

void TimeSliceCursor::advance(bool forward) {
    assert(crt_ts_ % precision_time_ == 0);
    assert(min_allowed_ts_ <= crt_ts_ && crt_ts_ <= max_allowed_ts_);

    if (forward) {
        // Forward
        if (crt_ts_ == max_allowed_ts_)
            return;

        // Next ts multiple of precision time
        crt_ts_ = std::min(max_allowed_ts_, crt_ts_ + precision_time_);

        crt_it_cbegin_ = crt_it_cend_; // Previous end ts becomes new begin ts
        crt_it_cend_   = std::lower_bound(crt_it_cbegin_, event_buffer_.cend(), crt_ts_,
                                        [](const EventCD &ev, timestamp ts) { return ev.t < ts; });
    } else {
        // Backward
        if (crt_ts_ == min_allowed_ts_)
            return;

        // Previous ts multiple of precision time
        crt_ts_ = std::max(min_allowed_ts_, crt_ts_ - precision_time_);

        crt_it_cbegin_ = event_buffer_.cbegin();
        crt_it_cend_   = std::lower_bound(crt_it_cbegin_, crt_it_cend_, crt_ts_,
                                        [](const EventCD &ev, timestamp ts) { return ev.t < ts; });
    }
}

void TimeSliceCursor::go_to(timestamp ts) {
    // Set current ts to the previous multiple of precision time and clamp it between extreme values
    crt_ts_ = std::min(max_allowed_ts_, std::max(min_allowed_ts_, precision_time_ * (ts / precision_time_)));

    crt_it_cbegin_ = event_buffer_.cbegin();
    crt_it_cend_   = std::lower_bound(crt_it_cbegin_, event_buffer_.cend(), crt_ts_,
                                    [](const EventCD &ev, timestamp ts) { return ev.t < ts; });
}

void TimeSliceCursor::go_to_begin() {
    go_to(min_allowed_ts_);
}

void TimeSliceCursor::go_to_end() {
    go_to(max_allowed_ts_);
}

timestamp TimeSliceCursor::ts() const {
    return crt_ts_;
}

TimeSliceCursor::EventBufferConstIterator TimeSliceCursor::cbegin() const {
    return crt_it_cbegin_;
}

TimeSliceCursor::EventBufferConstIterator TimeSliceCursor::cend() const {
    return crt_it_cend_;
}

bool TimeSliceCursor::is_at_the_end() const {
    return crt_ts_ == max_allowed_ts_;
}

} // namespace Metavision
