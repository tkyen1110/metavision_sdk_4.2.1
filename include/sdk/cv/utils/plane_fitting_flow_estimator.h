/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_PLANE_FITTING_FLOW_ESTIMATOR_H
#define METAVISION_SDK_CV_PLANE_FITTING_FLOW_ESTIMATOR_H

#include <vector>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

template<typename T>
class MostRecentTimestampBufferT;

/// @brief Class computing the flow's component in the normal direction of an edge moving in a time surface
///
/// The flow is computed by selecting recent timestamp values in a time surface around a given location,
/// fitting a plane to these timestamps using linear least-squares and inferring the flow from
/// the plane's estimated parameters.
/// This class enables rejecting visual flow estimates based on two quality indicators. The first indicator is the plane
/// fitting error on the timestamps of the timesurface, which is checked to lie within a configured tolerance. The
/// second indicator, denoted spatial consistency, measures the consistency between the radius of the considered
/// neighborhood and the distance covered by the edge during the time period observed in the local timesurface. The
/// visual flow estimates the speed of the local edge and we can calculate the distance covered by the local edge
/// between the timestamp of the oldest event used for plane fitting and the center timestamp. The ratio between this
/// covered distance and the radius of the neighborhood can be seen as a quality indicator for the estimated visual
/// flow, and can be used to reject visual flow estimates when the spatial consistency ratio lies outside a configured
/// range.
class PlaneFittingFlowEstimator {
public:
    /// @brief Constructor
    /// @param radius Radius used to select timestamps in a time surface around a given location
    /// @param enable_flow_normalization Flag to indicate if the estimated flow should be normalized
    /// @param min_spatial_consistency_ratio Lower bound of the acceptable range for the spatial consistency ratio
    /// quality indicator. Pass a negative value to disable this test.
    /// @param max_spatial_consistency_ratio Upper bound of the acceptable range for the spatial consistency ratio
    /// quality indicator. Pass a negative value to disable this test.
    /// @param fitting_error_tolerance Tolerance used to accept visual flow estimates with low enough fitting error.
    /// Pass a negative value to disable this test.
    /// @param neighbor_sample_fitting_fraction Fraction used to determine how many timestamps from the timesurface
    /// neighborhood are used to fit the plane.
    PlaneFittingFlowEstimator(int radius = 3, bool enable_flow_normalization = false,
                              float min_spatial_consistency_ratio = -1, float max_spatial_consistency_ratio = -1,
                              timestamp fitting_error_tolerance = -1, float neighbor_sample_fitting_fraction = 0.3f);

    /// @brief Tries to estimate the visual flow at the given location
    /// @tparam T Type of the estimated flow, either float or double
    /// @param[in] pxts Input time surface
    /// @param[in] x Abscissa at which the flow is to be estimated
    /// @param[in] y Ordinate at which the flow is to be estimated
    /// @param[out] vx Flow's x component, expressed in pixels/s, if the estimation succeeds
    /// @param[out] vy Flow's y component, expressed in pixels/s, if the estimation succeeds
    /// @param[in] c Polarity at which timestamps are to be sampled. If the value is -1, the polarity is automatically
    /// determined by looking at the most recent timestamp at the given location
    /// @param[out] time_limit_ptr Optional parameter that contains the oldest timestamp used during the flow estimation
    /// if the estimation has succeeded
    /// @return True if the estimation has succeeded, false otherwise
    template<typename T>
    bool get_flow(const MostRecentTimestampBufferT<timestamp> &pxts, int x, int y, T &vx, T &vy, int c = 0,
                  timestamp *time_limit_ptr = nullptr) const;

private:
    const int radius_, diameter_, nevents_;
    const bool enable_flow_normalization_;
    const float min_spatial_consistency_ratio_, max_spatial_consistency_ratio_;
    const timestamp fitting_error_tolerance_sqr_;
    const bool reject_using_spatial_consistency_, reject_using_fitting_error_;
    mutable std::vector<timestamp> tmp_times_;
};

using NormalFlowEstimator
    [[deprecated("This alias is deprecated since version 4.1.0 and will be removed in future releases")]] =
        PlaneFittingFlowEstimator;

} // namespace Metavision

#include "metavision/sdk/cv/utils/detail/plane_fitting_flow_estimator_impl.h"

#endif // METAVISION_SDK_CV_PLANE_FITTING_FLOW_ESTIMATOR_H
