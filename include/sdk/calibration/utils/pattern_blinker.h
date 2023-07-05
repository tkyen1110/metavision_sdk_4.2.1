/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_PATTERN_BLINKER_H
#define METAVISION_SDK_CALIBRATION_PATTERN_BLINKER_H

#include <opencv2/core/mat.hpp>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Class that generates a blinking pattern by returning, at a fixed frequency, either a white image or the image
/// of the pattern.
class PatternBlinker {
public:
    /// @brief Constructor loading the pattern from a colored image
    /// @param pattern_img Pattern's image
    /// @param blinking_rate Time period between two consecutive blinks (in us)
    PatternBlinker(const cv::Mat &pattern_img, timestamp blinking_rate = 1e4);

    /// @brief Swaps the blinking frame between "All White" and "With a Pattern", provided that the timestamp is less
    /// recent than the refresh period.
    /// @param ts Timestamp
    /// @param output_img Output copy of the current blink/blank frame
    /// @return true if image was updated
    bool update_blinking_image(const timestamp ts, cv::Mat &output_img);

    /// @brief Gets the current state of the blinking frame, without updating it.
    /// @param output_img Output copy of the current blink/blank frame
    void get_current_blinking_image(cv::Mat &output_img);

    /// @brief Returns the size of the blinking frame
    cv::Size get_image_size();

private:
    timestamp blinking_rate_;
    timestamp last_ts_;
    bool is_blank_frame_;

    cv::Mat front_img_, back_img_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CALIBRATION_PATTERN_BLINKER_H
