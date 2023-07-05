/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_SPARSE_OPTICAL_FLOW_ALGORITHM_H
#define METAVISION_SDK_CV_SPARSE_OPTICAL_FLOW_ALGORITHM_H

#include <memory>
#include <sstream>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/cv/events/event_optical_flow.h"
#include "metavision/sdk/cv/configs/sparse_optical_flow_config.h"

namespace Metavision {

class SparseOpticalFlowAlgorithmInternal;

/// @brief Algorithm used to compute the optical flow of objects
/// @note This sparse optical flow approach tracks small edge-like features and estimates the full motion for these
/// features. This algorithm runs flow estimation only on tracked features, which helps remaining efficient even on high
/// event-rate scenes. However, it requires the presence of trackable features in the scene, and the tuning of the
/// feature detection and tracking stage can be relatively complex.
/// @see PlaneFittingFlowAlgorithm and TripletMatchingFlowAlgorithm algorithms for simpler methods, with the drawbacks
/// of estimating only the component of the flow along the edge normal and with higher cost on high event-rate scenes.
class SparseOpticalFlowAlgorithm {
public:
    /// @brief Constructor
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @param config Sparse optical flow's configuration
    SparseOpticalFlowAlgorithm(int width, int height, const SparseOpticalFlowConfig &config);

    ///@brief Destructor
    ~SparseOpticalFlowAlgorithm();

    void set_height_limit(uint32_t min_y, uint32_t max_y);

    /// @brief Applies the optical flow algorithm to the given input buffer storing the result in the output buffer
    /// @tparam InputIt Read-Only input event iterator type
    /// @tparam OutputIt Read-Write output event iterator type
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter
    /// @return Iterator pointing to the past-the-end event added in the output
    /// @note the available implementations provided are:
    /// - InputIt = std::vector<Event2d>::iterator, OutputIt = std::back_insert_iterator<std::vector<EventOpticalFlow>>
    /// - InputIt = std::vector<Event2d>::const_iterator, OutputIt =
    /// std::back_insert_iterator<std::vector<EventOpticalFlow>>
    /// - InputIt = std::vector<Event2d>::iterator, OutputIt = std::vector<EventOpticalFlow>::iterator
    /// - InputIt = std::vector<Event2d>::const_iterator, OutputIt = std::vector<EventOpticalFlow>::iterator
    /// - InputIt = const Event2d *, OutputIt = EventOpticalFlow *
    template<typename InputIt, typename OutputIt>
    OutputIt process_events(InputIt it_begin, InputIt it_end, OutputIt inserter);

private:
    std::unique_ptr<SparseOpticalFlowAlgorithmInternal> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_SPARSE_OPTICAL_FLOW_ALGORITHM_H
