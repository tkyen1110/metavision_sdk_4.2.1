/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_JET_MONITORING_ALGORITHM_CONFIG_H
#define METAVISION_SDK_ANALYTICS_JET_MONITORING_ALGORITHM_CONFIG_H

#include <opencv2/core/types.hpp>

#include "metavision/sdk/base/utils/log.h"
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Jet monitoring algorithm parameters
struct JetMonitoringAlgorithmConfig {
    /// @brief Enum listing 4 possible nozzle orientations in the image reference frame
    ///
    /// The angle between the nozzle reference frame and the image reference frame must be inside {0; +-90; 180}
    enum Orientation {
        Down  = 0, // Jets move vertically downwards in the image
        Up    = 1, // Jets move vertically upwards in the image
        Left  = 2, // Jets move horizontally to the left in the image
        Right = 3  // Jets move horizontally to the right in the image
    };

    Orientation nozzle_orientation = Down; ///< Nozzle orientation

    cv::Rect detection_roi; ///< Central ROI used to detect jets by identifying peaks in the event-rate

    // Processing slice parameters
    int time_step_us         = 0;
    int accumulation_time_us = 0;

    // Thresholds for counting events. We normalize the count wrt the duration of the slice by
    // using kev/s so the thresholds don't depend much on the slice duration used
    int th_up_kevps   = 0; // A jet is detected when the event rate goes above this value (in kev/s)
    int th_down_kevps = 0; // A jet is considered finished when the event rate goes below this value (in kev/s)

    // The thresholds are considered exceeded only after a certain amount of time (filtering effect)
    timestamp th_up_delay_us   = 0;
    timestamp th_down_delay_us = 0;

    // Basic validation
    bool is_valid() const {
        bool all_ok = true;

        // Detection thresholds-related
        if (th_up_kevps < 0) {
            MV_LOG_ERROR() << "th_up_kevps is" << th_up_kevps << "and must be >= 0.";
            all_ok = false;
        }
        if (th_down_kevps < 0) {
            MV_LOG_ERROR() << "th_down_kevps is" << th_down_kevps << "and must be >= 0.";
            all_ok = false;
        }
        if (th_up_delay_us < 0) {
            MV_LOG_ERROR() << "th_up_delay is" << th_up_delay_us << "and must be >= 0.";
            all_ok = false;
        }
        if (th_down_delay_us < 0) {
            MV_LOG_ERROR() << "th_down_delay is" << th_down_delay_us << "and must be >= 0.";
            all_ok = false;
        }

        // Slice-related
        if (accumulation_time_us < 1) {
            MV_LOG_ERROR() << "accumulation_time_us is" << accumulation_time_us << "and must be >= 1.";
            all_ok = false;
        }
        if (time_step_us < 1) {
            MV_LOG_ERROR() << "time_step_us is" << time_step_us << "and must be >= 1.";
            // Return, because there will be a division by time_step_us.
            return false;
        }
        if ((accumulation_time_us % time_step_us) != 0) {
            MV_LOG_ERROR() << "accumulation_time_us (" << accumulation_time_us << ") must be multiple of time_step_us ("
                           << time_step_us << ").";
            all_ok = false;
        }

        if (detection_roi.x < 0 || detection_roi.y < 0 || detection_roi.width <= 0 || detection_roi.height <= 0) {
            MV_LOG_ERROR() << "detection_roi (x y width height) is not valid:" << detection_roi.x << detection_roi.y
                           << detection_roi.width << detection_roi.height;
            all_ok = false;
        }

        return all_ok;
    }
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_JET_MONITORING_ALGORITHM_CONFIG_H