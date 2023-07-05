/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_ANTI_FLICKER_ALGORITHM_H
#define METAVISION_SDK_CV_ANTI_FLICKER_ALGORITHM_H

#include <limits>

#include "metavision/sdk/base/events/event2d.h"
#include "metavision/sdk/cv/configs/frequency_estimation_config.h"

namespace Metavision {

/// @brief Algorithm used to remove flickering events given a frequency interval
class AntiFlickerAlgorithm {
public:
    /// @brief Builds a new AntiFlickerAlgorithm object
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @param config Frequency estimation's configuration
    AntiFlickerAlgorithm(int width, int height, const FrequencyEstimationConfig &config);

    /// @brief Destructor
    ~AntiFlickerAlgorithm() = default;

    /// @brief Sets minimum frequency of the flickering interval
    /// @note The value given has to be strictly inferior to maximum frequency
    /// @param min_freq Minimum frequency of the flickering interval
    /// @return false if value could not be set (invalid value)
    bool set_min_freq(double min_freq);

    /// @brief Sets maximum frequency of the flickering interval
    /// @note The value given has to be strictly superior to minimum frequency
    /// @param max_freq Maximum frequency of the flickering interval
    /// @return false if value could not be set (invalid value)
    bool set_max_freq(double max_freq);

    /// @brief Sets filter's length
    /// @param filter_length Number of values in the output median filter
    /// @return false if value could not be set (invalid value)
    bool set_filter_length(unsigned int filter_length);

    /// @brief Sets the difference allowed between two periods to be considered the same
    /// @param diff_thresh Maximum difference allowed between two successive periods to be considered the same
    void set_difference_threshold(double diff_thresh);

    /// @brief Processes a buffer of events and outputs the non-flickering ones
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter
    /// @return Iterator pointing to the past-the-end event added in the output
    template<class InputIt, class OutputIt>
    OutputIt process_events(InputIt it_begin, InputIt it_end, OutputIt inserter);

private:
    using period_precision = float;

    /// @brief Processes a single event and indicates whether the event should be kept
    /// @param event Event to process
    /// @return true if the event has a frequency outside the flickering interval
    template<class InputEvent>
    bool do_keep_event(InputEvent event);

    struct PixData {
        std::uint8_t last_pol       = -1;
        std::uint8_t cur_count      = 0;
        timestamp burst_first_ts[2] = {timestamp(-1),
                                       timestamp(-1)}; //< Values used to show that the polarities are not initialized
        period_precision last_meas  = std::numeric_limits<period_precision>::max(); //< Value used to force a change
        bool is_period_valid        = false;
    };

    int width_;                    ///< Sensor's width
    int num_pix_;                  ///< Number of pixels
    std::uint8_t filter_length_;   ///< Number of values of the same period before outputting an event
    period_precision min_period_;  ///< Lower limit on the output period
    period_precision max_period_;  ///< Upper limit on the output period
    period_precision diff_thresh_; ///< Maximum difference ratio to consider a change of period
    std::vector<PixData> state_;   ///< Current state of the pixels
};

} // namespace Metavision

#include "metavision/sdk/cv/utils/detail/anti_flicker_algorithm_impl.h"

#endif // METAVISION_SDK_CV_ANTI_FLICKER_ALGORITHM_H
