/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef TIME_SLICE_CURSOR_H
#define TIME_SLICE_CURSOR_H

#include <vector>
#include <metavision/sdk/base/events/event_cd.h>

namespace Metavision {

/// @brief Class that allows to manage a timeslice of events for PsmAlgorithm
///
/// The class is instantiated from a large buffer of events. It could be an entire record, as long as it fits in the
/// memory.
///
/// Then, the class acts like the Metavision::Camera class and allows to iterate over these events through buffers, the
/// length and step of which can be updated dynamically. It's also possible to jump to a specific timestamp or to get
/// the previous buffer. Because of this flexibility, the user cannot simply subscribe to an event callback. Once buffer
/// parameters have been updated and the buffer has been manually moved to the desired location, the user can retrieve
/// the begin and end event-iterators through the methods TimeSliceCursor::cbegin and TimeSliceCursor::cend.
///
/// The buffer parameters determine a set of possible timeslice positions. They can be changed at any time.
///   - Timestep between two consecutive buffers
///   - Duration of an event buffer
///
/// Possible timeslice updates are:
///   - Move forward one step of time
///   - Go back a step of time
///   - Go to the beginning of the sequence
///   - Go to the end of the sequence
///   - Jump to a specific timestamp
///
/// Obviously if we modify the step or the duration of the buffer and then ask to access the next buffer, it will start
/// at a multiple of the new timestep and its duration will change as well
///
/// @code{.cpp}
/// TimeSliceCursor algo (events, step, duration, 1);
/// // Process all events
/// while (!algo.is_at_the_end()) {
///     algo.advance(true);
///     process(algo.ts(), algo.cbegin(), algo.cend());
/// }
///
/// // Change params, go to ts=1000 and process backwards 3 buffers
/// algo.update_parameters(2 * step, 200, 1);
/// algo.go_to(1000);
/// for (int k = 0; k < 3; k++) {
///     process(algo.ts(), algo.cbegin(), algo.cend());
///     algo.advance(false);
/// }
/// @endcode
class TimeSliceCursor {
public:
    /// @brief Type of const event iterator
    using EventBufferConstIterator = std::vector<Metavision::EventCD>::const_iterator;

    /// @brief Constructor
    /// @param event_buffer Buffer of events, that the class is going to take ownership of
    /// @param precision_time Time step when moving the timeslice
    /// @param buffer_duration Temporal length of the timeslice in terms of multiple of @p precision_time
    /// @param num_process_before_matching Process accumulation used in the PSM algorithm, which indicates
    /// the minimum delay after the last event required to detect the very last particle
    TimeSliceCursor(std::vector<EventCD> &&event_buffer, int precision_time, int buffer_duration,
                    int num_process_before_matching);

    /// @brief Sets new parameters
    /// @param precision_time Time step when moving the timeslice
    /// @param buffer_duration Temporal length of the timeslice in terms of multiple of @p precision_time
    /// @param num_process_before_matching Process accumulation used in the PSM algorithm, which indicates
    /// the minimum delay after the last event required to detect the very last particle
    void update_parameters(int precision_time, int buffer_duration, int num_process_before_matching);

    /// @brief Increments the timeslice by the time step, forward or backward in time
    /// @param forward Forward or backward in time
    void advance(bool forward);

    /// @brief Jumps to a specific timestamp @p ts , which will be rounded to the nearest multiple of the time step
    /// @param ts Timestamp to reach
    void go_to(timestamp ts);

    /// @brief Jumps to the beginnning of the sequence
    void go_to_begin();

    /// @brief Jumps to the end of the sequence
    void go_to_end();

    /// @brief Checks if the timeslice is at the end of the sequence
    /// @return true if it's at the end
    bool is_at_the_end() const;

    /// @brief Gets current timeslice position
    /// @return The upper bound timestamp
    timestamp ts() const;

    /// @brief Gets const iterator to the first event in the timeslice
    /// @return cbegin
    EventBufferConstIterator cbegin() const;

    /// @brief Gets const iterator to the past-end event in the timeslice
    /// @return cend
    EventBufferConstIterator cend() const;

private:
    // Events
    const std::vector<EventCD> event_buffer_; ///< Events
    const timestamp min_events_ts_;           ///< Lowest event-timestamp
    const timestamp max_events_ts_;           ///< Largest event-timestamp

    // Dynamic lower and upper bounds (depends on time parameters)
    timestamp min_allowed_ts_; ///< Lowest possible value for the timeslice end (multiple of precision_time_)
                               ///< It's lower than min_events_ts_ so that the first timeslice can be empty
    timestamp max_allowed_ts_; ///< Largest possible value for the timeslice end (multiple of precision_time_)
                               ///< It's larger than max_events_ts_ so that the last timeslice can trigger
                               ///< the detection of the very last particle

    // Time parameters
    int precision_time_;              ///< Time step when moving the timeslice
    int buffer_duration_;             ///< Temporal length of the timeslice in terms of multiple of @p precision_time
    int num_process_before_matching_; ///< Process accumulation used in the PSM algorithm, which indicates the minimum
                                      ///< delay after the last event required to detect the very last particle

    // Timeslice
    timestamp crt_ts_;                       ///< End timestamp of the current timeslice
    EventBufferConstIterator crt_it_cbegin_; ///< Iterator to the first event of the timeslice
    EventBufferConstIterator crt_it_cend_;   ///< Iterator to the past-end event of the timeslice
};

} // namespace Metavision

#endif // TIME_SLICE_CURSOR_H
