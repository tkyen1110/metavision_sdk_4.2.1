/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_HISTOGRAM_DRAWING_HELPER_H
#define METAVISION_SDK_ANALYTICS_HISTOGRAM_DRAWING_HELPER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <stdexcept>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Class that draws an histogram
class HistogramDrawingHelper {
public:
    /// @brief Default Constructor
    HistogramDrawingHelper() = default;

    /// @brief Constructor
    /// @param height Height of the image
    /// @param hist_bins_centers Centers of the histogram bins
    HistogramDrawingHelper(int height, const std::vector<float> &hist_bins_centers);

    ~HistogramDrawingHelper() = default;

    /// @brief Updates data to display
    /// @param output_img Output image
    /// @param hist_counts Counts of the histogram
    void draw(cv::Mat &output_img, const std::vector<unsigned int> &hist_counts);

    /// @brief Gets width of the generated image
    int get_width() const;

private:
    int width_, height_;
    int n_bars_;
    float step_y_;
    cv::Mat bin_centers_img_;
    cv::Rect bin_centers_roi_;
    cv::Rect bars_roi_;

    const int text_width_space_ = 5;
    const int bars_max_width_   = 20;

    // colors
    cv::Vec3b color_txt_ = cv::Vec3b(219, 226, 228);
    cv::Vec3b color_bg_  = cv::Vec3b(52, 37, 30);
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_HISTOGRAM_DRAWING_HELPER_H