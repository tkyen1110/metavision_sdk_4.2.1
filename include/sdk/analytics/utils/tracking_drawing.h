/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_TRACKING_DRAWING_H
#define METAVISION_SDK_ANALYTICS_TRACKING_DRAWING_H

#include <opencv2/opencv.hpp>

#include "metavision/sdk/core/utils/random_color_map.h"
#include "metavision/sdk/analytics/events/event_spatter_cluster.h"
#include "metavision/sdk/analytics/events/event_tracking_data.h"

namespace Metavision {
namespace detail {

/// @brief Draws spatter cluster events
///
/// Drawing function used to draw tracking results from the @ref SpatterTrackerAlgorithm
/// (i.e. @ref EventSpatterCluster). Results are drawn as bounding boxes with tracked objects' ids beside.
///
/// @param ev The events cluster to draw.
/// @param output The output image in which the tracking result will be drawn.
inline void draw(const EventSpatterCluster &ev, cv::Mat &output) {
    const cv::Scalar &color_bbx = COLORS[ev.id % N_COLORS];
    cv::rectangle(output, cv::Point2f(ev.x, ev.y), cv::Point2f(ev.x + ev.width, ev.y + ev.height), color_bbx, 2);
    if (ev.untracked_times > 0)
        cv::putText(output, "U_" + std::to_string(ev.id), cv::Point2f(ev.x + ev.width, ev.y + ev.height), 1, 1,
                    color_bbx);
    else
        cv::putText(output, std::to_string(ev.id), cv::Point2f(ev.x + ev.width, ev.y + ev.height), 1, 1, color_bbx);
}

/// @brief Draws tracking data events
///
/// Drawing function used to draw tracking results from the @ref TrackingAlgorithm
/// (i.e. @ref EventTrackingData). Results are drawn as bounding boxes with tracked objects' ids beside.
///
/// @param ev The tracking data event to draw
/// @param output The output image in which the tracking result will be drawn
inline void draw(const EventTrackingData &ev, cv::Mat &output) {
    const cv::Scalar &color_bbx = COLORS[ev.object_id_ % N_COLORS];

    const auto hw = ev.width_ / 2;
    const auto hh = ev.height_ / 2;

    const cv::Point2f start(static_cast<float>(ev.x_ - hw), static_cast<float>(ev.y_ - hh));
    const cv::Point2f end = start + cv::Point2f(static_cast<float>(ev.width_), static_cast<float>(ev.height_));

    cv::rectangle(output, start, end, color_bbx, 2);
    cv::putText(output, std::to_string(ev.object_id_), end, 1, 1, color_bbx);
}

} // namespace detail

/// @brief Generic function used to draw tracking results
///
/// Results are drawn as bounding boxes with tracked objects' ids beside.
///
/// @tparam InputIt Iterator type over a tracking result
/// @param ts Current timestamp
/// @param begin First tracking result to draw
/// @param end End of the tracking results buffer
/// @param output The output image in which the tracking result will be drawn
template<typename InputIt>
void draw_tracking_results(timestamp ts, InputIt begin, InputIt end, cv::Mat &output) {
    // Add bounding boxes
    for (auto it = begin; it != end; ++it)
        Metavision::detail::draw(*it, output);

    // Display the current time
    const cv::Vec3b color_on(236, 223, 216);
    cv::putText(output, "Time: " + std::to_string(ts) + "us", cv::Point(10, 15), cv::FONT_HERSHEY_PLAIN, 1, color_on);
}

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_TRACKING_DRAWING_H
