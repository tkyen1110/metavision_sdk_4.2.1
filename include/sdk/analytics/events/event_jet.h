/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_EVENT_JET_H
#define METAVISION_SDK_ANALYTICS_EVENT_JET_H

#include <iostream>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Struct representing a detected jet event
struct EventJet {
public:
    /// @brief Jet number
    long count{0};

    /// @brief Timestamp of the beginning of the jet, in us
    timestamp t{-1};

    /// @brief Time since the beginning of the last jet, in us
    ///
    /// A negative value means this time-difference isn't defined yet because so far there has only been at most one jet
    timestamp previous_jet_dt{-1};

    /// @brief Default constructor
    EventJet() = default;

    /// @brief Constructor
    /// @param ts Timestamp of the beginning of the jet, in us
    /// @param count Jet number
    /// @param previous_jet_dt Jet duration, in us
    inline EventJet(timestamp ts, long count, timestamp previous_jet_dt) :
        t(ts), count(count), previous_jet_dt(previous_jet_dt) {}
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_EVENT_JET_H
