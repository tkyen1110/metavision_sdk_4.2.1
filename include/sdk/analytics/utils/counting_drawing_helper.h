/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_COUNTING_DRAWING_HELPER_H
#define METAVISION_SDK_ANALYTICS_COUNTING_DRAWING_HELPER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <stdexcept>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Class that superimposes line counting results on events
class CountingDrawingHelper {
public:
    /// @brief Default Constructor
    CountingDrawingHelper() = default;

    /// @brief Constructor
    /// @param line_counters_ordinates Ordinates of the lines tracker in the sensor's image
    CountingDrawingHelper(const std::vector<int> &line_counters_ordinates);

    ~CountingDrawingHelper() = default;

    /// @brief Adds a new line counter ordinate to the line_counters vector
    /// @param row Line ordinate
    void add_line_counter(int row);

    /// @brief Updates data to display
    /// @param ts Current timestamp
    /// @param count Last object count
    /// @param output_img Output image
    void draw(const timestamp ts, int count, cv::Mat &output_img);

private:
    /// @brief Displays all accumulated events
    void draw_line_counters(cv::Mat &output_img);

    /// @brief Displays the timestamp and the global count
    void draw_count(const timestamp ts, int count, cv::Mat &output_img);

    std::vector<int> line_counters_ordinates_;

    // colors
    cv::Vec3b color_txt_          = cv::Vec3b(219, 226, 228);
    cv::Vec3b color_line_tracker_ = cv::Vec3b(118, 114, 255);

    // counts
    int last_count_ = 0;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_COUNTING_DRAWING_HELPER_H
