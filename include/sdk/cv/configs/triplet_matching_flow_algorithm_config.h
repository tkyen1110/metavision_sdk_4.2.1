/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <cstdint>
#include "metavision/sdk/base/utils/timestamp.h"

#ifndef METAVISION_SDK_CV_TRIPLET_MATCHING_FLOW_ALGORITHM_CONFIG_H
#define METAVISION_SDK_CV_TRIPLET_MATCHING_FLOW_ALGORITHM_CONFIG_H

namespace Metavision {

/// @brief Structure representing the configuration of the triplet matching algorithm.
struct TripletMatchingFlowAlgorithmConfig {
    /// @brief Default constructor
    TripletMatchingFlowAlgorithmConfig() = default;

    /// @brief Initializing constructor
    /// @param radius Spatial radius to search for event matches
    /// @param dt_min Minimum time difference for event matches
    /// @param dt_max Maximum time difference for event matches
    TripletMatchingFlowAlgorithmConfig(float radius, timestamp dt_min, timestamp dt_max) :
        radius(radius), dt_max(dt_max), dt_min(dt_min) {}

    /// @brief Initializing constructor
    /// @param radius Spatial radius to search for event matches
    /// @param min_flow_mag Minimum flow magnitude to be observed, will be converted to a constraint on matches time
    /// difference
    /// @param max_flow_mag Maximum flow magnitude to be observed, will be converted to a constraint on matches time
    /// difference
    TripletMatchingFlowAlgorithmConfig(float radius, float min_flow_mag, float max_flow_mag) :
        radius(radius), dt_max(1e6f * radius / min_flow_mag), dt_min(1e6f / max_flow_mag) {}

    float radius;     ///< Matching spatial search radius. Higher values robustify and regularize estimated
                      ///< visual flow, but increase search area and number of potential matches, hence reduce
                      ///< efficiency. Note that this radius relates to the spatial search for event matching,
                      ///< but that the flow is actually estimated from 2 matches, so potentially twice this area.
    timestamp dt_max; ///< Matching temporal upper bound. Higher values lower minimum observable visual flow,
                      ///< but increase search area and number of potential matches, hence reduce efficiency.
    timestamp dt_min; ///< Matching temporal lower bound. Higher values reduce influence of noise when searching
                      ///< for matches in the past, by better filtering lateral observations of the same edge,
                      ///< but reduce the maximum observable visual flow.
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_TRIPLET_MATCHING_FLOW_ALGORITHM_CONFIG_H
