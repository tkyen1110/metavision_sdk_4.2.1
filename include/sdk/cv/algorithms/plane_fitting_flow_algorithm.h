/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_PLANE_FITTING_FLOW_ALGORITHM_H
#define METAVISION_SDK_CV_PLANE_FITTING_FLOW_ALGORITHM_H

#include "metavision/sdk/core/utils/mostrecent_timestamp_buffer.h"
#include "metavision/sdk/cv/utils/plane_fitting_flow_estimator.h"

namespace Metavision {

/// @brief This class is an optimized implementation of the dense optical flow approach proposed in Benosman R.,
/// Clercq C., Lagorce X., Ieng S. H., & Bartolozzi C. (2013). Event-based visual flow. IEEE transactions on neural
/// networks and learning systems, 25(2), 407-417.
/// @note This dense optical flow approach estimates the flow along the edge's normal, by fitting a plane locally in the
/// time-surface. The plane fitting helps regularize the estimation, but estimated flow results are still relatively
/// sensitive to noise. The algorithm is run for each input event, generating a dense stream of flow events, but making
/// it relatively costly on high event-rate scenes.
/// @see TripletMatchingFlowAlgorithm algorithm for a more efficient but more noise sensitive dense optical flow
/// approach.
/// @see SparseOpticalFlowAlgorithm algorithm for a flow algorithm based on sparse feature tracking, estimating the full
/// scene motion, staged hence more efficient on high event-rate scenes, but also more complex to tune and dependent on
/// the presence of trackable features in the scene.
class PlaneFittingFlowAlgorithm {
public:
    /// @brief Constructor
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @param radius Radius used to select timestamps in a time surface around a given location
    /// @param normalized_flow_magnitude Normalized magnitude of the optical flow. Pass a negative value or 0 to disable
    /// normalization.
    /// @param min_spatial_consistency_ratio Lower bound of the acceptable range for the spatial consistency ratio
    /// quality indicator. Pass a negative value to disable this test.
    /// @param max_spatial_consistency_ratio Upper bound of the acceptable range for the spatial consistency ratio
    /// quality indicator. Pass a negative value to disable this test.
    /// @param fitting_error_tolerance Tolerance used to accept visual flow estimates with low enough fitting error.
    /// Pass a negative value to disable this test.
    /// @param neighbor_sample_fitting_fraction Fraction used to determine how many timestamps from the timesurface
    /// neighborhood are used to fit the plane.
    PlaneFittingFlowAlgorithm(int width, int height, int radius = 3, float normalized_flow_magnitude = 100,
                              float min_spatial_consistency_ratio = -1, float max_spatial_consistency_ratio = -1,
                              timestamp fitting_error_tolerance = -1, float neighbor_sample_fitting_fraction = 0.3f) :
        pxts_(height, width, 2),
        estimator_(radius, normalized_flow_magnitude > 0, min_spatial_consistency_ratio, max_spatial_consistency_ratio,
                   fitting_error_tolerance, neighbor_sample_fitting_fraction),
        normalized_flow_magnitude_(normalized_flow_magnitude > 0 ? normalized_flow_magnitude : 1) {
        pxts_.set_to(0);
    }

    /// @brief Applies the get flow function to the given input buffer storing the result in the output buffer
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of @ref EventCD
    /// or equivalent
    /// @param[in] it_begin Iterator to first input event
    /// @param[in] it_end Iterator to the past-the-end event
    /// @param[out] inserter Output iterator or back inserter
    /// @return Iterator pointing to the past-the-end event added in the output
    template<typename InputIt, typename OutputIt>
    OutputIt process_events(InputIt it_begin, InputIt it_end, OutputIt inserter) {
        for (auto it = it_begin; it != it_end; ++it) {
            pxts_.at(it->y, it->x, it->p) = it->t;
            float vx, vy;
            if (estimator_.get_flow(pxts_, it->x, it->y, vx, vy, it->p)) {
                *inserter = EventOpticalFlow(it->x, it->y, it->p, it->t, vx * normalized_flow_magnitude_,
                                             vy * normalized_flow_magnitude_, id_++, it->x, it->y);
                ++inserter;
            }
        }

        return inserter;
    }

private:
    MostRecentTimestampBufferT<timestamp> pxts_;
    PlaneFittingFlowEstimator estimator_;
    int id_ = 0;
    const float normalized_flow_magnitude_;
};

using NormalFlowAlgorithm
    [[deprecated("This alias is deprecated since version 4.1.0 and will be removed in future releases")]] =
        PlaneFittingFlowAlgorithm;

} // namespace Metavision

#endif // METAVISION_SDK_CV_PLANE_FITTING_FLOW_ALGORITHM_H