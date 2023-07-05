/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_TRAIL_FILTER_ALGORITHM_H
#define METAVISION_SDK_CV_TRAIL_FILTER_ALGORITHM_H

#include <limits>
#include <memory>
#include <stdint.h>
#include <assert.h>

#include "metavision/sdk/core/algorithms/detail/internal_algorithms.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/base/events/event2d.h"

namespace Metavision {

/// @brief Filter that accepts an event either if the last event at the same coordinates was of different polarity,
/// or if it happened at least a given amount of time after the last event.
template<typename TTimestampType>
class TrailFilterAlgorithmT {
public:
    using TimestampType = TTimestampType;

public:
    /// @brief Builds a new @ref TrailFilterAlgorithmT object
    /// @param width Maximum X coordinate of the events in the stream
    /// @param height Maximum Y coordinate of the events in the stream
    /// @param threshold Length of the time window for activity filtering (in us)
    inline TrailFilterAlgorithmT(uint16_t width, uint16_t height, TTimestampType threshold);

    /// @brief Default destructor
    ~TrailFilterAlgorithmT() = default;

    /// @brief Applies the Trail filter to the given input buffer storing the result in the output buffer.
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref Event2d
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of @ref Event2d
    /// or equivalent
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter
    /// @return Iterator pointing to the past-the-end event added in the output
    template<class InputIt, class OutputIt>
    inline OutputIt process_events(InputIt it_begin, InputIt it_end, OutputIt inserter) {
        MV_SDK_LOG_DEBUG() << "TrailFilterAlgorithm::process() with threshold:" << threshold_;
        return Metavision::detail::insert_if(it_begin, it_end, inserter, std::ref(*this));
    }

    /// @brief Returns the current threshold
    /// @return Threshold applied while filtering
    inline TTimestampType get_threshold() const;

    /// @brief Set the current threshold
    /// @param threshold Current threshold
    inline void set_threshold(TTimestampType threshold);

    /// @brief Check if the event is filtered
    /// @return true if the event is filtered
    inline bool operator()(const Event2d &event);

    bool is_accepted(const Event2d &ev) {
        return this->operator()(ev);
    }

    bool is_rejected(const Event2d &ev) {
        return !this->operator()(ev);
    }

private:
    struct stat {
        struct coded_state {
            TTimestampType last_pol_ : 1;
            TTimestampType last_ts_pad_ : sizeof(TTimestampType) * 8 - 1;
        };
        union {
            TTimestampType last_ts_;
            coded_state anonymous;
        };
    };

    const uint16_t width_;
    const uint16_t height_;
    TTimestampType threshold_ = 0;

    std::vector<stat> states_;
};

template<typename TTimestampType>
inline TrailFilterAlgorithmT<TTimestampType>::TrailFilterAlgorithmT(uint16_t width, uint16_t height,
                                                                    TTimestampType threshold) :
    width_(width), height_(height), threshold_(threshold), states_(width_ * height) {
    for (auto &element : states_) {
        element.last_ts_            = std::numeric_limits<TTimestampType>::max() - 2 * threshold_;
        element.anonymous.last_pol_ = 0;
    }

    // The timestamp representation assumes that last_pol_
    // will be the lowest order bits
    // so that (ev.t >> timeshift) - ptr->last_ts_ is a valid comparison
    // and ptr->last_ts_ is updated through ptr->anonymous.pad = timestamp >> 1;
    //
    // However, the bit field order is compiler-dependent,
    // see https://en.cppreference.com/w/cpp/language/bit_field
    // check the assumption explicitly once
    static bool runtime_check_once = true;
    if (runtime_check_once) {
        runtime_check_once = false;
        stat check_assumption;
        check_assumption.last_ts_ = 1 << 1;
        if (check_assumption.anonymous.last_ts_pad_ != 1) {
            throw std::logic_error("Invalid bit field order");
        }
    }
}

template<typename TTimestampType>
inline bool TrailFilterAlgorithmT<TTimestampType>::operator()(const Event2d &event) {
    assert(event.x < width_);
    assert(event.y < height_);

    auto &state = states_[event.y * width_ + event.x];

    const auto p           = event.p;
    const TTimestampType t = static_cast<TTimestampType>(event.t);

    // ------------------------------
    // Accepted if last pol has changed or if the event has the same polarity but occurs x us after the last event
    // triggered on the same pixel in the trail
    const auto accepted = (p != state.anonymous.last_pol_) || (t - (state.last_ts_) >= threshold_);
    if (accepted) {
        // update if and only if the event is accepted
        state.anonymous.last_pol_ = p;
    }
    state.anonymous.last_ts_pad_ = t >> 1; // We shift by 1 to fit in the bitfield. We then lose 1 bit of precision (2
                                           // microseconds). But when we read last_ts via the union, we gain 1 bit (as
                                           // last_pol_ is a low bit) to remain on 32 bits. The overall operation is
                                           // equivalent to assign last_ts_ to event.t & 0xFFFFFFFE.
    return accepted;
}

template<typename TTimestampType>
typename TrailFilterAlgorithmT<TTimestampType>::TimestampType
    TrailFilterAlgorithmT<TTimestampType>::get_threshold() const {
    return threshold_;
}

template<typename TTimestampType>
void TrailFilterAlgorithmT<TTimestampType>::set_threshold(TTimestampType threshold) {
    threshold_ = threshold;
}

/// @brief Instantiation of the @ref TrailFilterAlgorithmT class using 32 bits unsigned integers
using TrailFilterAlgorithm = TrailFilterAlgorithmT<uint32_t>;

} // namespace Metavision

#endif // METAVISION_SDK_CV_TRAIL_FILTER_ALGORITHM_H
