/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_FREQUENCY_ALGORITHM_H
#define METAVISION_SDK_CV_FREQUENCY_ALGORITHM_H

#include <memory>

#include "metavision/sdk/cv/configs/frequency_estimation_config.h"

namespace Metavision {

class FrequencyAlgorithmInternal;

/// @brief Algorithm used to estimate the flickering frequency (Hz) of the pixels of the sensor
class FrequencyAlgorithm {
public:
    using frequency_precision = float;
    using period_precision    = float;

    /// @brief Builds a new FrequencyAlgorithm object
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @param frequency_config Frequency estimation's configuration
    FrequencyAlgorithm(int width, int height, const FrequencyEstimationConfig &frequency_config);

    /// @brief Destructor
    ~FrequencyAlgorithm();

    /// @brief Sets minimum frequency to output
    /// @note The value given has to be < maximum frequency
    /// @param min_freq Minimum frequency to output
    /// @return false if value could not be set (invalid value)
    bool set_min_freq(double min_freq);

    /// @brief Sets maximum frequency to output
    /// @note The value given has to be > minimum frequency
    /// @param max_freq Maximum frequency to output
    /// @return false if value could not be set (invalid value)
    bool set_max_freq(double max_freq);

    /// @brief Sets filter filter length
    /// @param filter_length Number of values in the output median filter
    /// @return false if value could not be set (invalid value)
    bool set_filter_length(unsigned int filter_length);

    /// @brief Sets the difference allowed between two periods to be considered the same
    /// @param diff_thresh Maximum difference allowed between two successive periods to be considered the same
    void set_difference_threshold(period_precision diff_thresh);

    /// @brief Processes a buffer of events and output inserter for Event2dFrequency<float>
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of
    /// @ref Event2dFrequency
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter for Event2dFrequency<float>
    template<class InputIt, class OutputIt>
    void process_events(InputIt it_begin, InputIt it_end, OutputIt inserter);

private:
    std::unique_ptr<FrequencyAlgorithmInternal> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_FREQUENCY_ALGORITHM_H
