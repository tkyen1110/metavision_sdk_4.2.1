/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_EVENT_PERIOD_H
#define METAVISION_SDK_CV_EVENT_PERIOD_H

#include "metavision/sdk/base/events/detail/event_traits.h"
#include "metavision/sdk/base/utils/detail/struct_pack.h"
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Event2dPeriod is an Event2d extended with a period information.
template<typename T = timestamp>
struct Event2dPeriod {
public:
    /// @brief Column position in the sensor at which the event happened
    unsigned short x;

    /// @brief Row position in the sensor at which the event happened
    unsigned short y;

    /// @brief Polarity of the event
    /// @sa @ref Metavision::Event2d for more details
    short p;

    /// @brief Timestamp at which the event happened (in us)
    timestamp t;

    T period;

    Event2dPeriod() {}

    Event2dPeriod(unsigned short x, unsigned short y, timestamp t, T period) : x(x), y(y), t(t), period(period) {}

    void write_event(void *buf, timestamp origin) const {
        RawEvent *buffer = static_cast<RawEvent *>(buf);
        buffer->ts       = t - origin;
        buffer->x        = x;
        buffer->y        = y;
        buffer->period   = period;
    }

    FORCE_PACK(
        /// Structure of size 64 bits to represent one event
        struct RawEvent {
            uint32_t ts;
            unsigned int x : 14;
            unsigned int y : 14;
            uint32_t period;
        });
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_EVENT_PERIOD_H
