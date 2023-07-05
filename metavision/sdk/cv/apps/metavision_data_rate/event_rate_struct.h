/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_EVENT_RATE_STRUCT_H
#define METAVISION_SDK_EVENT_RATE_STRUCT_H

#include <opencv2/opencv.hpp>

namespace Metavision {

/// @brief Class representing an event used to describe an event rate
struct EventRateStruct {
public:
    /// @brief Default constructor
    EventRateStruct() = default;

    /// @brief Constructor
    /// @param rate Event rate (in kEv/s)
    /// @param t Timestamp of the event (in us)
    EventRateStruct(int rate, timestamp t) : rate(rate), t(t) {}

    /// @brief Timestamp at which the event happened (in us)
    timestamp t;

    /// @brief Event rate (in kEv/s)
    int rate;
};

} // namespace Metavision

#endif // METAVISION_SDK_EVENT_RATE_STRUCT_H
