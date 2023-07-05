/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_JET_MONITORING_DRAWING_HELPER_H
#define METAVISION_SDK_ANALYTICS_JET_MONITORING_DRAWING_HELPER_H

#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <stdexcept>

#include "metavision/sdk/base/utils/timestamp.h"

#include "metavision/sdk/analytics/configs/jet_monitoring_algorithm_config.h"

namespace Metavision {

/// @brief Class that superimposes jet monitoring results on events
class JetMonitoringDrawingHelper {
public:
    /// @brief Constructor
    /// @param camera_roi Region of interest used by the camera (Left x, Top y, width, height)
    /// @param jet_roi Region of interest used by the jet-monitoring algorithm to detect jets (Left x, Top y, width,
    /// height)
    /// @param nozzle_orientation Nozzle orientation
    JetMonitoringDrawingHelper(const cv::Rect &camera_roi, const cv::Rect &jet_roi,
                               const JetMonitoringAlgorithmConfig::Orientation &nozzle_orientation);

    ~JetMonitoringDrawingHelper() = default;

    /// @brief Updates data to display
    /// @param ts Current timestamp
    /// @param count Last object count
    /// @param er_kevps Event rate in k-ev per second
    /// @param output_img Output image
    void draw(const timestamp ts, int count, int er_kevps, cv::Mat &output_img);

private:
    const cv::Rect camera_roi_;
    const cv::Rect jet_roi_;
    cv::Rect bg_noise_roi_1_;
    cv::Rect bg_noise_roi_2_;

    cv::Point arrow_pt1_; ///< The point the arrow starts from
    cv::Point arrow_pt2_; ///< The point the arrow points to
    const int arrow_length_ = 20;

    const double font_scale_ = 1.8;

    const cv::Vec3b color_txt_          = cv::Vec3b(219, 226, 228);
    const cv::Vec3b color_roi_          = cv::Vec3b(118, 114, 255);
    const cv::Vec3b color_bg_noise_roi_ = cv::Vec3b(201, 126, 64);
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_JET_MONITORING_DRAWING_HELPER_H
