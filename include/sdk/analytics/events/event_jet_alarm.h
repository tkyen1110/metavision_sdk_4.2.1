/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_EVENT_JET_ALARM_H
#define METAVISION_SDK_ANALYTICS_EVENT_JET_ALARM_H

#include <string>
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Struct representing a jet monitoring alarm
struct EventJetAlarm {
public:
    /// @brief Event timestamp; timestamp at which the event was generated, in us
    timestamp t{-1};

    /// @brief Alarm condition timestamp; timestamp at which the alarm condition occurred, in us
    /// @note This is always <= @ref t because of the confirmation delays in the algorithm
    timestamp alarm_ts{-1};

    /// @brief Alarm types
    enum struct AlarmType {
        JetNotDetected,
        JetTooEarly,
        TooManyJets,
    } alarm_type;

    /// @brief Detailed information on the alarm
    std::string info;

    /// @brief Default constructor
    EventJetAlarm() = default;

    /// @brief Constructor
    /// @param ts Timestamp of the jet, in us
    /// @param alarm_type Alarm type
    inline EventJetAlarm(timestamp ts, AlarmType alarm_type) : t(ts), alarm_type(alarm_type) {}
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_EVENT_JET_ALARM_H
