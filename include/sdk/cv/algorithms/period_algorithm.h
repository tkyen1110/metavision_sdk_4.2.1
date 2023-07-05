/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_PERIOD_ALGORITHM_H
#define METAVISION_SDK_CV_PERIOD_ALGORITHM_H

#include <vector>
#include <iterator>
#include <memory>
#include <sstream>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/cv/configs/period_estimation_config.h"

namespace Metavision {

template<typename T>
class PeriodAlgorithmInternal;

template<typename T>
class Event2dPeriod;

/// @brief Algorithm used to estimate the flickering period of the pixels of the sensor
class PeriodAlgorithm {
public:
    /// Timestamp precision
    using period_precision = float;

    /// @brief Builds a new PeriodAlgorithm object
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @param config period estimation's configuration
    PeriodAlgorithm(int width, int height, const PeriodEstimationConfig &config);

    /// @brief Destructor
    ~PeriodAlgorithm();

    /// @brief Sets minimum period to output
    /// @note The value @p min_period has to be smaller than the maximum period
    /// @param min_period Minimum period (us) to output
    /// @return false if value could not be set (invalid value)
    bool set_min_period(double min_period);

    /// @brief Sets maximum period to output
    /// @note The value @p max_period has to be larger than the minimum period
    /// @param max_period Maximum period to output
    /// @return false if value could not be set (invalid value)
    bool set_max_period(double max_period);

    /// @brief Sets filter filter length
    /// @param filter_length Number of values in the output median filter
    /// @return false if value could not be set (invalid value)
    bool set_filter_length(unsigned int filter_length);

    /// @brief Sets the difference allowed between two periods to be considered the same
    /// @param diff_thresh Maximum difference allowed between two successive periods to be considered the same
    void set_difference_threshold(period_precision diff_thresh);

    /// @brief Processes a buffer of events and output inserter for Event2dPeriod<T>
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of
    /// @ref Event2dPeriod
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter for Event2dPeriod<T>
    template<class InputIt, class OutputIt>
    void process_events(InputIt it_begin, InputIt it_end, OutputIt inserter);

private:
    std::unique_ptr<PeriodAlgorithmInternal<period_precision>> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_PERIOD_ALGORITHM_H
