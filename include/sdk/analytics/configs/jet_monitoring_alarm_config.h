/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_JET_MONITORING_ALARM_CONFIG_H
#define METAVISION_SDK_ANALYTICS_JET_MONITORING_ALARM_CONFIG_H

#include "metavision/sdk/base/utils/log.h"
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Jet monitoring alarm parameters
struct JetMonitoringAlarmConfig {
    /// @brief If true, an alarm will be raised if the jet count *exceeds* the expected_count value
    bool alarm_on_count = false;
    /// @brief Maximum expected number of jets
    int max_expected_count = 0;

    /// @brief Activate/deactivate alarm on cycle time
    bool alarm_on_cycle = false;
    /// @brief Expected cycle time
    float expected_cycle_ms = 0;
    /// @brief Tolerance for estimated cycle time, in percentage of @ref expected_cycle_ms
    float cycle_tol_percentage = 0;

    /// @brief Basic validation
    bool is_valid() const {
        bool all_ok = true;

        if (alarm_on_count) {
            if (max_expected_count < 0) {
                MV_LOG_ERROR() << "expected_count is" << max_expected_count << "and must be >= 0.";
                all_ok = false;
            }
        }

        if (alarm_on_cycle) {
            if (expected_cycle_ms <= 0.f) {
                MV_LOG_ERROR() << "expected_cycle_ms is" << expected_cycle_ms << "and must be > 0.";
                all_ok = false;
            }
            if (cycle_tol_percentage < 0.1f) {
                MV_LOG_ERROR() << "cycle_tol_percentage is" << cycle_tol_percentage << "and must be >= 0.1.";
                all_ok = false;
            }
        }

        return all_ok;
    }
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_JET_MONITORING_ALARM_CONFIG_H