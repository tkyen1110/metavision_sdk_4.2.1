/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_TRIPLET_MATCHING_FLOW_ALGORITHM_H
#define METAVISION_SDK_CV_TRIPLET_MATCHING_FLOW_ALGORITHM_H

#include <array>
#include <memory>
#include <vector>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/cv/configs/triplet_matching_flow_algorithm_config.h"
#include "metavision/sdk/cv/events/event_optical_flow.h"

namespace Metavision {

class TripletMatchingFlowAlgorithmInternal;

/// @brief This class implements the dense optical flow approach proposed in Shiba S., Aoki Y., & Gallego G. (2022).
/// "Fast Event-Based Optical Flow Estimation by Triplet Matching". IEEE Signal Processing Letters, 29, 2712-2716.
/// @note This dense optical flow approach estimates the flow along the edge's normal, by locally searching for aligned
/// events triplets. The flow is estimated by averaging all aligned event triplets found, which helps regularize the
/// estimates, but results are still relatively sensitive to noise. The algorithm is run for each input event,
/// generating a dense stream of flow events, but making it relatively costly on high event-rate scenes.
/// @see PlaneFittingFlowAlgorithm algorithm for slightly more accurate but more expensive dense optical flow approach.
/// @see SparseOpticalFlowAlgorithm algorithm for a flow algorithm based on sparse feature tracking, estimating the full
/// scene motion, staged hence more efficient on high event-rate scenes, but also more complex to tune and dependent on
/// the presence of trackable features in the scene.
class TripletMatchingFlowAlgorithm {
public:
    using Config = TripletMatchingFlowAlgorithmConfig;

    /// @brief Constructor
    /// @param width Sensor width
    /// @param height Sensor height
    /// @param config Configuration for triplet matching flow algorithm, @ref TripletMatchingFlowAlgorithmConfig
    TripletMatchingFlowAlgorithm(int width, int height, const Config &config);

    ///@brief Destructor
    ~TripletMatchingFlowAlgorithm();

    /// @brief Applies the optical flow algorithm to the given input buffer storing the result in the output buffer
    /// @tparam InputIt Read-Only input event iterator type on EventCD objects
    /// @tparam OutputIt Read-Write output event iterator type on EventOpticalFlow objects
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter
    /// @return Iterator pointing to the past-the-end event added in the output
    template<typename InputIt, typename OutputIt>
    OutputIt process_events(InputIt it_begin, InputIt it_end, OutputIt inserter);

private:
    std::unique_ptr<TripletMatchingFlowAlgorithmInternal> pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_TRIPLET_MATCHING_FLOW_ALGORITHM_H
