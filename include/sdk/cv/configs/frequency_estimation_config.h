/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_FREQUENCY_ESTIMATION_CONFIG_H
#define METAVISION_SDK_CV_FREQUENCY_ESTIMATION_CONFIG_H

namespace Metavision {

/// @brief Configuration used when estimating the frequency of periodic signals
struct FrequencyEstimationConfig {
    /// @brief Constructor
    /// @param filter_length Number of measures of the same period before outputting an event
    /// @param min_freq Minimum frequency to output.
    /// @param max_freq Maximum frequency to output.
    /// @param diff_thresh_us Maximum difference (us) allowed between two consecutive periods to be considered the same.
    /// @note Internally, the algorithm first estimates period using the events' timestamps and then converts it
    /// to frequency. The decision-making criterion @p diff_thresh_us is then absolute in the period domain, instead
    /// of the frequency domain.
    /// @param output_all_burst_events Whether all the events of a burst must be output or not
    FrequencyEstimationConfig(unsigned int filter_length = 7, float min_freq = 10.f, float max_freq = 150.f,
                              unsigned int diff_thresh_us = 1500, bool output_all_burst_events = false) :
        filter_length_(filter_length),
        min_freq_(min_freq),
        max_freq_(max_freq),
        diff_thresh_us_(diff_thresh_us),
        output_all_burst_events_(output_all_burst_events) {}

    unsigned int filter_length_;
    float min_freq_;
    float max_freq_;
    unsigned int diff_thresh_us_;
    bool output_all_burst_events_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_FREQUENCY_ESTIMATION_CONFIG_H
