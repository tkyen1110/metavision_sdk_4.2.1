/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_DFT_HIGH_FREQ_SCORER_ALGORITHM_H
#define METAVISION_SDK_CALIBRATION_DFT_HIGH_FREQ_SCORER_ALGORITHM_H

#include <opencv2/core/core.hpp>
#include <memory>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/calibration/configs/dft_high_freq_scorer_algorithm_config.h"

namespace Metavision {

class DftHighFreqScorerAlgorithmInternal;

/// @brief Class that computes and returns the Discrete Fourier Transform (DFT) of an image in addition to a score
/// depending on the proportion of high frequencies.
class DftHighFreqScorerAlgorithm {
public:
    /// @brief Constructor
    /// @param width Sensor's width (in pixels)
    /// @param height Sensor's height (in pixels)
    /// @param dft_config Discrete Fourier Transform configuration file
    DftHighFreqScorerAlgorithm(int width, int height, const DftHighFreqScorerAlgorithmConfig &dft_config);

    ~DftHighFreqScorerAlgorithm();

    /// @brief Computes the DFT and a high frequency score
    /// @param ts Timestamp
    /// @param input_frame Binary frame of accumulated blinking events (Either 0 or 1)
    /// @param output_score Ratio of the average DFT magnitude of high frequencies to the one for all frequencies
    /// @return false if it fails
    bool process_frame(const timestamp ts, const cv::Mat &input_frame, float &output_score);

private:
    std::unique_ptr<DftHighFreqScorerAlgorithmInternal> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CALIBRATION_DFT_HIGH_FREQ_SCORER_ALGORITHM_H
