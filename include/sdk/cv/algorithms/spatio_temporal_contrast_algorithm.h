/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_SPATIO_TEMPORAL_CONTRAST_ALGORITHM_H
#define METAVISION_SDK_CV_SPATIO_TEMPORAL_CONTRAST_ALGORITHM_H

#include <memory>
#include <limits>
#include <vector>

#include "metavision/sdk/core/algorithms/detail/internal_algorithms.h"
#include "metavision/sdk/base/utils/detail/bitinstructions.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/base/events/event2d.h"

namespace Metavision {
/// @brief The SpatioTemporalContrast Filter is a noise filter using the exponential response of a pixel
/// to a change of light to filter out wrong detections and trails.
///
/// For an event to be forwarded, it needs to be preceded by another one in a given time window, this
/// ensures that the spatio temporal contrast detection is strong enough.
/// It is also possible to then cut all the following events up to a change of polarity in the stream for
/// that particular pixel (strong trail removal). Note that this will remove signal if 2 following
/// edges of the same polarity are detected (which should not happen that frequently).
///
/// @note The timestamp may be stored in different types 64 bits, 32 bits or 16 bits.
/// The behavior may vary from one size to the other since the number of significant bits may change.
/// Before using the version with less than 32 bits check that the behavior is still valid for the usage.
template<class itype>
class SpatioTemporalContrastAlgorithmT;

/// @brief Instantiation of the @ref SpatioTemporalContrastAlgorithmT with 32 bits unsigned integers
using SpatioTemporalContrastAlgorithm = SpatioTemporalContrastAlgorithmT<uint32_t>;

/// @brief Instantiation of the @ref SpatioTemporalContrastAlgorithmT with 16 bits unsigned integers
using SmallSpatioTemporalContrastAlgorithm = SpatioTemporalContrastAlgorithmT<uint16_t>;

template<class itype>
class SpatioTemporalContrastAlgorithmT {
public:
    /// @brief Builds a new SpatioTemporalContrast object
    /// @param width Maximum X coordinate of the events in the stream
    /// @param height Maximum Y coordinate of the events in the stream
    /// @param threshold Length of the time window for filtering (in us)
    /// @param cut_trail If true, after an event goes through, it removes all events until change of polarity
    SpatioTemporalContrastAlgorithmT(int width, int height, timestamp threshold, bool cut_trail = true) :
        timeshift(0), pixels_(MAX_WIDTH * height), not_cut_trail_(!cut_trail), stats_(pixels_) {
        set_threshold(threshold);

        if (width > MAX_WIDTH) {
            MV_SDK_LOG_ERROR() << "SpatioTemporalContrastAlgorithmT can accepts only" << std::to_string(MAX_WIDTH)
                               << "as max width.";
            std::abort();
        }
        for (auto &element : stats_) {
            element.last_ts_                      = std::numeric_limits<itype>::max() - 2 * thres_;
            element.anonymous.last_pol_           = 0;
            element.anonymous.last_pol_change_ok_ = not_cut_trail_;
        }

        // The timestamp representation assumes that last_pol_ and last_pol_change_ok_
        // will be the lowest order bits
        // so that (ev.t >> timeshift) - ptr->last_ts_ is a valid comparison
        // and ptr->last_ts_ is updated through ptr->anonymous.pad = timestamp >> 2;
        //
        // However, the bit field order is compiler-dependent,
        // see https://en.cppreference.com/w/cpp/language/bit_field
        // check the assumption explicitly once
        static bool runtime_check_once = true;
        if (runtime_check_once) {
            runtime_check_once = false;
            stat check_assumption;
            check_assumption.last_ts_ = 1 << 2;
            if (check_assumption.anonymous.pad != 1) {
                throw std::logic_error("Invalid bit field order");
            }
        }
    }
    ~SpatioTemporalContrastAlgorithmT() = default;

    bool operator()(const Event2d &ev) {
        unsigned int index = ((ev.y) << BITS_FOR_WIDTH) + (ev.x);
        auto ptr           = &stats_[index];
        // Are we in the window from the latest event at this pixel location?
        auto const p = ev.p;
        itype t      = static_cast<itype>(ev.t >> timeshift);
        itype dt     = static_cast<itype>(ev.t >> timeshift) - ptr->last_ts_;
        if (ptr->anonymous.last_pol_ == p) {
            if (ptr->anonymous.last_pol_change_ok_ && dt <= thres_) {
                // Forward event only if we are not cutting trail or the polarity changed from last event

                ptr->anonymous.last_pol_change_ok_ = not_cut_trail_;
                ptr->anonymous.pad                 = t >> 2;
                return true;
            }
        } else {
            ptr->anonymous.last_pol_change_ok_ = 1;
            ptr->anonymous.last_pol_           = p;
        }
        ptr->anonymous.pad = t >> 2;
        return false;
    }

    /// @brief Processes a buffer of events and outputs filtered events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter
    /// @return Iterator pointing to the past-the-end event added in the output
    template<typename InputIt, typename OutputIt>
    OutputIt process_events(InputIt it_begin, InputIt it_end, OutputIt inserter) {
        MV_SDK_LOG_DEBUG() << "SpatioTemporalContrastAlgorithmT::process_output() with threshold:"
                           << (thres_ << timeshift);
        return Metavision::detail::insert_if(
            it_begin, it_end, inserter, std::ref(*this),
            Metavision::detail::Prefetch((const char *)&stats_[0], MAX_WIDTH * sizeof(stat)));
    }

    /// @brief Returns the threshold for STC filtering
    /// @return Threshold applied while STC filtering (in us)
    timestamp get_threshold() {
        return static_cast<timestamp>(thres_ << timeshift);
    }

    /// @brief Sets the threshold for STC filtering
    /// @param threshold Length of the time window for STC filtering (in us)
    void set_threshold(timestamp threshold) {
        if (DO_TIMESHIFT) {
            int last_bit                 = 0;
            int first_bit                = 0;
            unsigned int shift_threshold = static_cast<unsigned int>(threshold >> 2);
            if (shift_threshold != 0) {
                last_bit  = sizeof(shift_threshold) * 8 - clz(shift_threshold);
                first_bit = ctz(shift_threshold);
            }
            // Value evaluated from the result of the blinking_pattern_focus command
            // The goal is to keep enough significant bits for the threshold and
            // the mask
            timeshift = std::max(last_bit - MASKBITS / 2 - 2, 0);
            if (first_bit < timeshift) {
                MV_SDK_LOG_ERROR() << Log::no_space << "STC threshold (" << threshold
                                   << ") may not behave as expected. To fit in memory limit: " << MASKBITS
                                   << " the threshold is shift by: " << timeshift
                                   << ". Some threshold bits are not considered. Last bit: " << last_bit;
            }
        }
        thres_ = static_cast<itype>(threshold >> timeshift);
        MV_SDK_LOG_DEBUG() << "SpatioTemporalContrastAlgorithmT::set_threshold() with threshold:"
                           << (thres_ << timeshift) << "Shifting the time by:" << timeshift;
    }

    /// @brief Returns the cut_trail parameter for STC filtering
    /// @return The cut_trail parameter for STC filtering used for STC filtering
    bool get_cut_trail() {
        return !not_cut_trail_;
    }

    /// @brief Sets the cut_trail parameter for STC filtering
    /// @param v If true, after an event goes through, it removes all events until change of polarity
    void set_cut_trail(bool v) {
        not_cut_trail_ = !v;
    }

    bool is_accepted(const Event2d &ev) {
        return this->operator()(ev);
    }

    bool is_rejected(const Event2d &ev) {
        return !this->operator()(ev);
    }

private:
    static constexpr int MASKBITS = sizeof(itype) * 8 - 2;
    struct stat {
        struct coded_state {
            itype last_pol_ : 1;
            itype last_pol_change_ok_ : 1;
            itype pad : MASKBITS;
        };
        union {
            itype last_ts_;
            coded_state anonymous;
        };
    };

    int timeshift;
    int pixels_;
    itype thres_;
    bool not_cut_trail_;
    std::vector<stat> stats_;
    static constexpr int BITS_FOR_WIDTH = 11;
    static constexpr int MAX_WIDTH      = 1 << (BITS_FOR_WIDTH);
    static constexpr bool DO_TIMESHIFT  = MASKBITS > 22 ? false : true;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_SPATIO_TEMPORAL_CONTRAST_ALGORITHM_H
