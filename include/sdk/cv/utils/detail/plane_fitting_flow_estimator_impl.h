/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_PLANE_FITTING_FLOW_ESTIMATOR_IMPL_H
#define METAVISION_SDK_CV_PLANE_FITTING_FLOW_ESTIMATOR_IMPL_H

#include "metavision/sdk/core/utils/mostrecent_timestamp_buffer.h"
#include "metavision/sdk/cv/events/event_optical_flow.h"

namespace Metavision {

template<typename real_t>
bool PlaneFittingFlowEstimator::get_flow(const MostRecentTimestampBufferT<timestamp> &pxts, int x, int y, real_t &vx,
                                         real_t &vy, int c, timestamp *time_limit_ptr) const {
    static_assert(std::is_floating_point<real_t>::value, "Non floating-point output type!");

    if (x < radius_ || x >= pxts.cols() - radius_ || y < radius_ || y >= pxts.rows() - radius_) {
        return false;
    }

    // If no polarity is provided, use the most recent one at the given (x,y) coordinate
    if (c < 0) {
        const auto pbuff_ts_max = std::max_element(pxts.ptr(y, x), pxts.ptr(y, x) + pxts.channels());
        c                       = pbuff_ts_max - pxts.ptr(y, x);
    }

    // Determine the time limit allowing to keep only the most recent nevents_ events
    tmp_times_.clear();
    const int stride     = (pxts.cols() - diameter_) * pxts.channels();
    const auto *pbuff_ts = pxts.ptr(y - radius_, x - radius_, c);
    for (int dy = 0; dy < diameter_; ++dy, pbuff_ts += stride) {
        for (int dx = 0; dx < diameter_; ++dx, pbuff_ts += pxts.channels()) {
            tmp_times_.push_back(pbuff_ts[0]);
        }
    }
    std::nth_element(tmp_times_.begin(), tmp_times_.begin() + nevents_, tmp_times_.end(), std::greater<timestamp>());
    const auto time_limit = *(tmp_times_.begin() + nevents_);
    if (time_limit_ptr != nullptr)
        *time_limit_ptr = time_limit;

    if (time_limit <= 0) {
        return false;
    }

    // We want to fit a plane of equation t = a*x + b*y to the selected timestamp values in the neighborhood
    // The plane-fitting cost function is: cost(a,b) = sum_i (a*x_i+b*y_i-t_i)²
    // Minimum is obtained for grad_cost(a,b) = 0
    // => [a; b] = inv([sum_i x_i², sum_i x_i*y_i; sum_i x_i*y_i, sum_i y_i²]) * [sum_i t_i*x_i; sum_i t_i*y_i]
    // Time-surface gradient is grad[t(x,y)] = [delta_t/delta_x; delta_t/delta_y] = [a; b]
    // The visual flow has the same direction as the time-surface gradient
    // and its magnitude is the inverse of the time-surface slope in the direction
    // of the edge normal, i.e. the inverse of the magnitude of the time-surface gradient.

    // Collect intermediate values needed to compute the optimal plane
    const auto t_ctr = pxts.at(y, x, c);
    auto sum_x2 = 0, sum_y2 = 0;
    auto sum_xy = 0;
    auto sum_xt = 0, sum_yt = 0;
    real_t sum_t2 = 0;
    pbuff_ts      = pxts.ptr(y - radius_, x - radius_, c);
    for (int y = -radius_; y <= radius_; ++y, pbuff_ts += stride) {
        const int y2     = y * y;
        auto local_sum_t = 0, local_sum_x = 0;
        for (int x = -radius_; x <= radius_; ++x, pbuff_ts += pxts.channels()) {
            if (pbuff_ts[0] < time_limit)
                continue;
            const int delta_ts = pbuff_ts[0] - t_ctr;
            sum_x2 += x * x;
            sum_y2 += y2;
            local_sum_x += x;
            sum_xt += x * delta_ts;
            local_sum_t += delta_ts;
            sum_t2 += delta_ts * delta_ts;
        }
        sum_xy += y * local_sum_x;
        sum_yt += y * local_sum_t;
    }

    // Solve least-square problem
    const int determinant = std::abs(sum_x2 * sum_y2 - sum_xy * sum_xy);
    if (determinant == 0) {
        return false;
    }
    const real_t inv_determinant = real_t(1) / determinant;
    const real_t a               = (sum_y2 * sum_xt - sum_xy * sum_yt) * inv_determinant;
    const real_t b               = (sum_x2 * sum_yt - sum_xy * sum_xt) * inv_determinant;
    const real_t grad_norm2      = a * a + b * b;
    if (grad_norm2 <= 0) {
        return false;
    }
    const real_t grad_norm = std::sqrt(grad_norm2);

    // Check quality indicators
    const real_t inv_flow_norm = grad_norm;
    if (reject_using_spatial_consistency_) {
        const float spatial_consistency_ratio = std::abs(radius_ * inv_flow_norm / (t_ctr - time_limit));
        if (spatial_consistency_ratio < min_spatial_consistency_ratio_)
            return false;
        if (spatial_consistency_ratio > max_spatial_consistency_ratio_)
            return false;
    }
    if (reject_using_fitting_error_) {
        const real_t cost_solution =
            (a * a * sum_x2 + b * b * sum_y2 + sum_t2 + 2 * a * b * sum_xy - 2 * b * sum_xt - 2 * a * sum_yt) /
            nevents_;
        if (cost_solution > fitting_error_tolerance_sqr_) {
            return false;
        }
    }

    // Compute and return the estimated visual flow
    if (enable_flow_normalization_) {
        const real_t scale = 1 / grad_norm; // Result has no unit
        vx                 = a * scale;
        vy                 = b * scale;
    } else {
        const real_t scale = real_t(1e6) / grad_norm2; // Result is in pixels / s
        vx                 = a * scale;
        vy                 = b * scale;
    }
    return true;
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_PLANE_FITTING_FLOW_ESTIMATOR_IMPL_H
