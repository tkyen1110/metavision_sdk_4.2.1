/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_DETAIL_EDGELET_2D_TRACKING_ALGORITHM_IMPL_H
#define METAVISION_SDK_CV3D_DETAIL_EDGELET_2D_TRACKING_ALGORITHM_IMPL_H

#include <opencv2/opencv.hpp>

#include "metavision/sdk/cv3d/utils/edgelet_utils.h"

namespace Metavision {

template<typename Edgelet2dIt, typename StatusIt, typename OutputEdgelet2dIt>
OutputEdgelet2dIt Edgelet2dTrackingAlgorithm::process(const MostRecentTimestampBuffer &time_surface, timestamp target,
                                                      Edgelet2dIt edgelet_begin, Edgelet2dIt edgelet_end,
                                                      OutputEdgelet2dIt d_begin, StatusIt status_begin) {
    // Ideally we want both directions to be collinear, but we allow a 30° deviation angle
    // Minimum cosine between the edgelet direction and the tracked direction for the edgelet to be tracked
    // (i.e. cos(30°))
    constexpr float MIN_DIR2_DOT_MAIN_DIRECTION = .866f;

    // Minimum number of matches around the median to consider a match as valid
    constexpr int MIN_MEDIAN_MATCHES = 2;

    // If the number of points is even, let's put the edgelet's center in the middle
    const float n_support_points = static_cast<float>(params_.n_support_points_);
    const float half_n_support_points =
        (params_.n_support_points_ % 2) ? std::floor(n_support_points / 2) : (n_support_points - 1) / 2;

    const float xymin = float(params_.search_radius_);
    const float xmax  = float(time_surface.cols() - params_.search_radius_);
    const float ymax  = float(time_surface.rows() - params_.search_radius_);

    // Process the input 2D edgelets
    // status_begin will always be updated (i.e. one for each input edgelet), while d_begin will only be updated in case
    // of a match.
    for (auto it = edgelet_begin; it != edgelet_end; ++it, ++status_begin) {
        matches_.clear();
        matches_idx_.clear();
        match_candidates_.clear();

        // Creates a support point every PIXEL_SIZE and tracks it
        for (int i = 0; i < static_cast<int>(params_.n_support_points_); ++i) {
            const auto signed_distance = i - half_n_support_points;
            cv::Matx21f suppt_img = it->ctr2_ + params_.support_points_distance_ * signed_distance * it->unit_dir2_;

            if (suppt_img(0) < xymin || suppt_img(1) < xymin || suppt_img(0) >= xmax || suppt_img(1) >= ymax) {
                continue;
            }

            cv::Matx21f match;
            timestamp match_ts;
            int match_idx;

            if (track_support_point_both_directions(time_surface, target - params_.threshold_, suppt_img,
                                                    it->unit_norm2_, params_.search_radius_, match_candidates_,
                                                    match_ts, match_idx)) {
                matches_.push_back(match_candidates_[match_idx]);
                matches_idx_.push_back(match_idx);
            }
        }

        // If not enough matches --> tracking failed
        if (matches_.size() < 2 || matches_.size() < params_.n_support_points_ / 2) {
            *status_begin = false;
            continue;
        }

        // Compute the median of the tracking distance to filter wrong matches
        // Warning: this is not the true median in case of even numbers
        const int n_median = matches_idx_.size() / 2;
        std::nth_element(matches_idx_.begin(), matches_idx_.begin() + n_median, matches_idx_.end());
        const int median_idx = matches_idx_[n_median];

        // Found additional edges -> compute new normal and center using a Least Square
        cv::Matx22f H      = cv::Matx22f::zeros();
        cv::Matx21f center = cv::Matx21f::zeros();

        int n_val   = 0;
        float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0, sum_yy = 0;
        using SizeType = std::vector<cv::Matx21f>::size_type;
        for (SizeType i = 0; i < matches_.size(); ++i) {
            if (std::abs(matches_idx_[i] - median_idx) <= params_.median_outlier_threshold_) {
                center += matches_[i];

                const auto &x = matches_[i](0);
                const auto &y = matches_[i](1);

                sum_x += x;
                sum_xx += x * x;
                sum_y += y;
                sum_yy += y * y;
                sum_xy += x * y;

                ++n_val;
            }
        }

        if (n_val < MIN_MEDIAN_MATCHES) {
            *status_begin = false;
            continue;
        }

        const auto one_over_n_val = 1.f / static_cast<float>(n_val);

        center *= one_over_n_val;
        sum_x *= one_over_n_val;
        sum_y *= one_over_n_val;
        sum_xy *= one_over_n_val;
        sum_xx *= one_over_n_val;
        sum_yy *= one_over_n_val;

        H(0, 0) = sum_xx - sum_x * sum_x;
        H(0, 1) = sum_xy - sum_x * sum_y;
        H(1, 0) = H(0, 1);
        H(1, 1) = sum_yy - sum_y * sum_y;

        cv::Matx21f eig_val;
        cv::Matx22f eig_vec;
        cv::eigen(H, eig_val, eig_vec);

        // Check the angle between the tracked direction and the edgelet direction
        const float dir2_dot_main_direction = it->unit_dir2_.dot(eig_vec.get_minor<1, 2>(0, 0).t());

        if (std::abs(dir2_dot_main_direction) < MIN_DIR2_DOT_MAIN_DIRECTION) {
            *status_begin = false;
            continue;
        }

        const float sign            = dir2_dot_main_direction > 0 ? 1.f : -1.f;
        const cv::Matx21f direction = sign * eig_vec.get_minor<1, 2>(0, 0).t();

        // Compute the coordinates of the barycenter of the tracked points in the frame formed by the normal of the
        // edgelet and the tracked edgelet's direction. This frame is well defined (can't collapse into a 1D frame)
        // because these two vectors can't be colinear.
        const cv::Matx22f transfer_mat = {it->unit_norm2_(0), -direction(0), it->unit_norm2_(1),
                                          -direction(1)}; // Transfer matrix between the two bases
        const cv::Matx21f center_shift = center - it->ctr2_;
        const cv::Matx21f coordinates  = transfer_mat.inv() * center_shift; // Retrieve the coordinates in the new base

        // Estimate the new center using only the coordinate of the center shift along the normal axis to limit the
        // drift along the direction of the edgelet
        const cv::Matx21f matched_center = it->ctr2_ + coordinates(0) * it->unit_norm2_;

        *d_begin++ = {matched_center, direction, edgelet_normal_from_direction(direction)};

        *status_begin = true;
    }

    return d_begin;
}
} // namespace Metavision

#endif // METAVISION_SDK_CV3D_DETAIL_EDGELET_2D_TRACKING_ALGORITHM_IMPL_H
