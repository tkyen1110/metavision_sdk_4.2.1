/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_PERIOD_ESTIMATION_CONFIG_H
#define METAVISION_SDK_CV_PERIOD_ESTIMATION_CONFIG_H

namespace Metavision {

/// @brief Configuration used when estimating the period of periodic signals
struct PeriodEstimationConfig {
    /// @brief Constructor
    /// @param filter_length Number of measures of the same period before outputting an event
    /// @param min_period Minimum period (us) to output. Default value is 6500us, which corresponds to 153Hz
    /// @param max_period Maximum period (us) to output. Default value is 1e5us, which corresponds to 10Hz
    /// @param diff_thresh_us Maximum difference (us) allowed between two consecutive periods to be considered the same
    /// @param output_all_burst_events Whether all the events of a burst must be output or not
    PeriodEstimationConfig(unsigned int filter_length = 7, float min_period = 6500, unsigned int max_period = 1e5,
                           float diff_thresh_us = 1500, bool output_all_burst_events = false) :
        filter_length_(filter_length),
        min_period_(min_period),
        max_period_(max_period),
        diff_thresh_us_(diff_thresh_us),
        output_all_burst_events_(output_all_burst_events) {}

    unsigned int filter_length_;
    float min_period_;
    float max_period_;
    unsigned int diff_thresh_us_;
    bool output_all_burst_events_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_PERIOD_ESTIMATION_CONFIG_H
