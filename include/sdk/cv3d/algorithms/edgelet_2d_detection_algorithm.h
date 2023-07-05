/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_EDGELET_2D_DETECTION_ALGORITHM_H
#define METAVISION_SDK_CV3D_EDGELET_2D_DETECTION_ALGORITHM_H

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/core/utils/mostrecent_timestamp_buffer.h"

namespace Metavision {

/// @brief Algorithm used to detect 2D edgelets in a time surface
class Edgelet2dDetectionAlgorithm {
public:
    /// @brief Constructor
    /// @param threshold Detection tolerance threshold
    /// @sa @ref is_fast_edge for more details
    Edgelet2dDetectionAlgorithm(timestamp threshold = 0) : threshold_(threshold) {}

    /// @brief Destructor
    ~Edgelet2dDetectionAlgorithm() = default;

    /// @brief Tries to detect 2D edgelets in the time surface at locations given by the input events
    /// @tparam InputIt Read-Only input event iterator type
    /// @tparam OutputIt Read-Write output @ref EventEdgelet2d event iterator type
    /// @param ts Time surface in which 2D edgelets are looked for
    /// @param begin First iterator to the buffer of events whose locations will be looked at to detect edgelets
    /// @param end Last iterator to the buffer of events whose locations will be looked at to detect edgelets
    /// @param d_begin Output iterator of 2D edgelets buffer
    /// @return Iterator pointing to the past-the-end event added in the output 2D edgelets buffer
    template<typename InputIt, typename OutputIt>
    OutputIt process(const MostRecentTimestampBuffer &ts, InputIt begin, InputIt end, OutputIt d_begin);

private:
    timestamp threshold_; ///< Tolerance threshold used in 2D edgelet detection
};

} // namespace Metavision

#include "metavision/sdk/cv3d/algorithms/detail/edgelet_2d_detection_algorithm_impl.h"

#endif // METAVISION_SDK_CV3D_EDGELET_2D_DETECTION_ALGORITHM_H
