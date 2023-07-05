/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_DFT_HIGH_FREQ_SCORER_ALGORITHM_CONFIG_H
#define METAVISION_SDK_CALIBRATION_DFT_HIGH_FREQ_SCORER_ALGORITHM_CONFIG_H

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Structure to instantiate a DftHighFreqScorerAlgorithm configuration.
struct DftHighFreqScorerAlgorithmConfig {
    /// @brief Constructor
    /// @param refresh_period_us Time period between two consecutive process (Skip the blinking frames that are too
    /// close in time to the last one processed)
    /// @param use_inverted_gray Invert the gray levels so that white becomes black (and conversely)
    /// If true, map linearly 0 to 255 and 255 to 0; if false, keep the pixel intensity.
    /// @param speed_up Skip the process if the frame hasn't changed
    /// @param high_pass_filter_dft_rows_ratio Ratio of the image's height determining the radius for the High Pass
    /// Filter on the DFT (Ex: if ratio = 1/8 and (width,height) = (640,480), then radius = 480/8 = 60 for the filter)
    DftHighFreqScorerAlgorithmConfig(timestamp refresh_period_us = 1e4, bool use_inverted_gray = false,
                                     bool speed_up = false, double high_pass_filter_dft_rows_ratio = 1.0 / 8) :
        refresh_period_us_(refresh_period_us),
        use_inverted_gray_(use_inverted_gray),
        speed_up_(speed_up),
        high_pass_filter_dft_rows_ratio_(high_pass_filter_dft_rows_ratio) {}

    timestamp refresh_period_us_;
    bool use_inverted_gray_;
    bool speed_up_;
    double high_pass_filter_dft_rows_ratio_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CALIBRATION_DFT_HIGH_FREQ_SCORER_ALGORITHM_CONFIG_H
