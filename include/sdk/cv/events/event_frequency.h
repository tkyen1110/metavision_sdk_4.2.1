/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_EVENT_FREQUENCY_H
#define METAVISION_SDK_CV_EVENT_FREQUENCY_H

#include <cstdint>

#include "metavision/sdk/base/events/detail/event_traits.h"
#include "metavision/sdk/base/utils/detail/struct_pack.h"
#include "metavision/sdk/base/utils/timestamp.h"

using timestamp = Metavision::timestamp;

namespace Metavision {

/// @brief Event2dFrequency represents a periodic event at a particular 2D position.
template<typename T = float>
struct Event2dFrequency {
    static_assert(std::is_floating_point<T>::value,
                  "Event2dFrequency can only be instantiated with a floating point type.");

public:
    /// @brief Column position in the sensor at which the event happened
    unsigned short x;

    /// @brief Row position in the sensor at which the event happened
    unsigned short y;

    /// @brief Timestamp at which the event happened (in us)
    timestamp t;

    /// @brief Frequency
    T frequency;

    Event2dFrequency() {}
    Event2dFrequency(unsigned short x, unsigned short y, timestamp t, T frequency) :
        x(x), y(y), t(t), frequency(frequency) {}
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_EVENT_FREQUENCY_H
