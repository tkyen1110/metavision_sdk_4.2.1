/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_EDGELET_UTILS_H
#define METAVISION_SDK_CV3D_EDGELET_UTILS_H

#include <opencv2/core.hpp>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/core/utils/mostrecent_timestamp_buffer.h"

namespace Metavision {

class PlaneFittingFlowEstimator;

/// @brief Computes the 2D edgelet's direction from its normal
/// @tparam Ti The input vector type for the normal
/// @tparam To The output vector type for the direction
/// @param[in] normal The 2D edgelet's normal
/// @return The edgelet's direction
template<typename Ti, typename To = Ti>
inline To edgelet_direction_from_normal(const Ti &normal);

/// @brief Computes the 2D edgelet's normal from its direction
/// @tparam Ti The input vector type for the direction
/// @tparam To The output vector type for the normal
/// @param[in] direction The 2D edgelet's direction
/// @return The edgelet's normal
template<typename Ti, typename To = Ti>
inline To edgelet_normal_from_direction(const Ti &direction);

/// @brief Tries to detect a 2D edgelet at the specified event's location
///
/// To detect a 2D edgelet, the algorithm samples timestamp values along a discrete circle around the event's location,
/// and checks that half of the circle has timestamps inferior to the other half.
///
/// The 2D edgelet is looked for in the time surface's channel corresponding to the event's polarity.
///
/// @param[in] ts Time surface in which an 2D edgelet is looked for
/// @param[in] evt Event whose polarity and location will be used to detect an 2D edgelet
/// @param[in] threshold Tolerance threshold meaning that a timestamp on the circle's lowest half should be inferior to
/// the lowest timestamp on the circle's highest half plus this value
/// @param[out] normal If a 2D edgelet is found and this parameter provided, it corresponds to the detected 2D edgelet's
/// normal
/// @return true if a 2D edgelet has been detected at the event's location, False otherwise
/// @note Adapted the code released from "Mueggler, E., Bartolozzi, C., & Scaramuzza, D. (2017). Fast event-based corner
/// detection" to detect edges rather than corners
bool is_fast_edge(const MostRecentTimestampBuffer &ts, const EventCD &evt, timestamp threshold = 0,
                  cv::Matx21f *normal = nullptr);

/// @brief Tries to match an edgelet's support point (i.e. a point sampled along an edgelet) in a time surface
///
/// Matching candidates are sampled along the direction of the edgelet's normal and within a given search radius. A
/// support point is then matched to a matching candidate when its timestamp is more recent than a target one.
///
/// @param[in] time_surface Time surface in which matches are looked for
/// @param[in] ts_target Oldest timestamp allowed for a valid match
/// @param[in] pt2_img Coordinates of the support point in the time surface
/// @param[in] vec2_img_search Search direction (i.e. edgelet's normal)
/// @param[in] search_radius Radius in which matches are looked for
/// @param[out] match_candidates Coordinates of the matching candidates that will be tested.
/// This buffer will be filled by this function.
/// @param[out] ts_match Matched candidate's timestamp
/// @param[out] match_idx If the tracking succeeds, it will contain the index of the match in the matching candidates
/// buffer
/// @return true if the tracking has succeeded, false otherwise
bool track_support_point_both_directions(const MostRecentTimestampBuffer &time_surface, const timestamp &ts_target,
                                         const cv::Matx21f &pt2_img, const cv::Matx21f &vec2_img_search,
                                         unsigned int search_radius, std::vector<cv::Matx21f> &match_candidates,
                                         timestamp &ts_match, int &match_idx);

/// @brief Tries to track a support point on a slope generated by a moving edge having a known orientation
///
/// Matching candidates are sampled along the direction of the edgelet's normal and within a given search radius. A
/// support point is then matched to a matching candidate when the latter lies on a slope generated by an edge having
/// the same orientation as the expected one. Edge orientations are quantized into 4 directions (0°-180°), (45°-225°),
/// (90°-270°) and (135°-315°).
///
/// @param[in] time_surface Time surface in which matches are looked for
/// @param[in] flow_estimator Estimator used to determine the slope on which matching candidates are lying
/// @param[in] pt2_img Coordinates of the support point in the time surface
/// @param[in] vec2_img_dir Expected edge's direction
/// @param[in] search_radius Radius in which matches are looked for
/// @param[out] match_candidates Coordinates of the matching candidates that have been tested
/// @param[out] ts_match Matched candidate's timestamp
/// @param[out] match_idx If the tracking succeeds, it will contain the index of the match in the matching candidates
/// buffer
/// @return true if the tracking has succeeded, false otherwise
bool track_support_point_on_expected_slope(const MostRecentTimestampBuffer &time_surface,
                                           const PlaneFittingFlowEstimator &flow_estimator, const cv::Matx21f &pt2_img,
                                           const cv::Matx21f &vec2_img_dir, unsigned int search_radius,
                                           std::vector<cv::Matx21f> &match_candidates, timestamp &ts_match,
                                           int &match_idx);

} // namespace Metavision

#include "metavision/sdk/cv3d/utils/detail/edgelet_utils_impl.h"

#endif // METAVISION_SDK_CV3D_EDGELET_UTILS_H
