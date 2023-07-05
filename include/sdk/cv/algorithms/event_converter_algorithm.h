/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_EVENT_CONVERTER_ALGORITHM_H
#define METAVISION_SDK_CV_EVENT_CONVERTER_ALGORITHM_H

#include <functional>

#include "metavision/sdk/core/algorithms/detail/internal_algorithms.h"

namespace Metavision {

/// This algorithm applies the `()` operator of the Converter class to the input.
template<typename Converter>
class EventConverterAlgorithm {
public:
    using event_type     = typename Converter::event_type;
    using converted_type = typename Converter::event_type;

    /// Constructor
    /// @param converter Instance of the Converter class
    constexpr EventConverterAlgorithm(const Converter &converter = Converter()) : converter_(converter) {}

    template<typename InputIt, typename OutputIt>
    constexpr void process(InputIt first, InputIt last, OutputIt d_first) {
        detail::transform(first, last, d_first, std::ref(*this));
    }

    constexpr converted_type operator()(const event_type &event) {
        return converter_(event);
    }

private:
    Converter converter_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_EVENT_CONVERTER_ALGORITHM_H
