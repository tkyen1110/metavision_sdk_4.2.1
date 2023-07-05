/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_TRANSPOSE_EVENTS_ALGORITHM_H
#define METAVISION_SDK_CV_TRANSPOSE_EVENTS_ALGORITHM_H

#include <memory>
#include <sstream>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/algorithms/detail/internal_algorithms.h"
#include "metavision/sdk/base/events/event2d.h"

namespace Metavision {

/// @brief Class that switches X and Y coordinates of an event stream.
/// This filter changes the dimensions of the corresponding frame (width and height are switched)
class TransposeEventsAlgorithm {
public:
    TransposeEventsAlgorithm(){};

    /// @brief Applies the Transpose filter to the given input buffer storing the result in the output buffer
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter
    /// @return Iterator pointing to the past-the-end event added in the output
    template<class InputIt, class OutputIt>
    inline OutputIt process_events(InputIt it_begin, InputIt it_end, OutputIt inserter) {
        return detail::transform(it_begin, it_end, inserter, std::ref(*this));
    }

    /// @brief Applies the Transpose filter to the given input buffer storing the result in the output buffer.
    /// @param ev Event2d to be updated
    inline void operator()(Event2d &ev) const;
};

inline void TransposeEventsAlgorithm::operator()(Metavision::Event2d &ev) const {
    std::swap(ev.x, ev.y);
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_TRANSPOSE_EVENTS_ALGORITHM_H
