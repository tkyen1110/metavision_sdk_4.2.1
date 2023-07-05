/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_RECTANGLE_ROI_UTILS_H
#define METAVISION_SDK_ANALYTICS_RECTANGLE_ROI_UTILS_H

#include <functional>
#include <array>

namespace Metavision {

/// @brief Struct representing a rectangular ROI
struct RectangleRoi {
    RectangleRoi() {}

    RectangleRoi(const int x, const int y, const int width, const int height) :
        x_min(x), x_max(x + width), y_min(y), y_max(y + height) {}

    int x_min, x_max;
    int y_min, y_max;
};

/// @brief Accumulates events corresponding to multiple ROIs
/// @tparam InputIt Iterator type of an event-buffer, the elements of which have x, y and p as attributes
/// @tparam RoiIt Iterator type of a buffer of @ref RectangleRoi
/// @tparam OutputIt Iterator type of a buffer of std::array<int, 2>
/// @param it_begin First iterator to a buffer of events
/// @param it_end Past-end iterator to a buffer of events
/// @param roi_begin First iterators to a buffer of ROIs
/// @param roi_end Past-end iterator to a buffer of ROIs
/// @param output Iterator to a buffer of std::array<int, 2>
template<typename InputIt, typename RoiIt, typename OutputIt>
void accumulate_events_rectangle_roi(InputIt it_begin, InputIt it_end, RoiIt roi_begin, RoiIt roi_end,
                                     OutputIt output) {
    for (auto it_roi = roi_begin; it_roi != roi_end; ++it_roi, ++output) {
        const RectangleRoi &roi   = *it_roi;
        std::array<int, 2> &count = *output;

        for (auto it = it_begin; it != it_end; ++it) {
            if (it->x >= roi.x_min && it->x < roi.x_max && it->y >= roi.y_min && it->y < roi.y_max) {
                ++count[it->p];
            }
        }
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_RECTANGLE_ROI_UTILS_H
