/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_JET_MONITORING_ALARMS_H
#define METAVISION_SDK_ANALYTICS_JET_MONITORING_ALARMS_H

#include <functional>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/analytics/events/event_jet.h"
#include "metavision/sdk/analytics/events/event_jet_alarm.h"
#include "metavision/sdk/analytics/configs/jet_monitoring_alarm_config.h"

namespace Metavision {

/// @brief Class that generates alarms depending on the results of the JetMonitoringAlgorithm
class JetMonitoringAlarms {
public:
    /// @brief Constructor
    /// @param params Alarm parameters
    /// @param jet_detection_delay_us The time that the jet monitoring algorithm takes to confirm a jet. This is needed
    /// so we don't generate an alarm for a missing jet when it has already started but not yet confirmed.
    JetMonitoringAlarms(const JetMonitoringAlarmConfig &params, timestamp jet_detection_delay_us);

    /// @brief Default destructor
    ~JetMonitoringAlarms() = default;

    using AlarmCallback = std::function<void(const EventJetAlarm &)>;

    /// @brief Reset internal state
    void reset_state();

    /// @brief Sets the callback that is called when an alarm is raised
    /// @param cb Callback processing a const reference of @ref EventJetAlarm
    void set_on_alarm_callback(const AlarmCallback &cb);

    /// @brief Processes a jet, and generates alarms if needed, by calling @p callback
    void process_jet(timestamp ts, const EventJet &jet);

    /// @brief Processes a slice and generates alarms if needed, by calling @p callback
    /// (the main purpose is to monitor if a jet has not arrived in time)
    void process_slice(timestamp ts, bool jet_is_present, timestamp ts_since_last_jet);

private:
    bool alarm_on_count_    = false;
    int max_expected_count_ = 0;

    bool alarm_on_cycle_           = false;
    timestamp min_jet_dt_us_       = 0;
    timestamp max_jet_dt_us_       = 0;
    timestamp jet_detection_delay_ = 0;

    /// @brief This indicates if an alarm of type "no-jet" has been triggered, so we don't trigger multiple alarms for
    /// the same missing jet.
    bool no_jet_alarm_triggered_ = false;

    /// @brief Callback called on each alarm
    AlarmCallback alarm_cb_{nullptr};
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_JET_MONITORING_ALARMS_H