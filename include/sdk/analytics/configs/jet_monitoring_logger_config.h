/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_JET_MONITORING_LOGGER_CONFIG_H
#define METAVISION_SDK_ANALYTICS_JET_MONITORING_LOGGER_CONFIG_H

#include "metavision/sdk/base/utils/log.h"

namespace Metavision {

/// @brief Jet monitoring logger parameters
struct JetMonitoringLoggerConfig {
    // Logging parameters (user-friendly representation, as the logger only cares about slices)
    bool enable_logging       = false; ///< If true, the logging is enabled, otherwise no logging
    int log_history_length_ms = 0;     ///< Log buffer length in ms. Each log dump will have this duration
    int log_dump_delay_ms     = 0;     ///< Log delay after the trigger in ms. How much we see after the trigger
    std::string log_out_dir   = "";    ///< Base output directory. Each log dump will be a subdirectory in it
    bool dump_at_exit         = false; ///< Automatically dump when exiting the application
    bool log_jet_video        = false; ///< Combine the images of each jet to create a video of an average jet
    bool log_jets_evt_rate    = false; ///< Dump a file containing the event rate for each jet

    // Basic validation.
    bool is_valid() const {
        bool all_ok = true;

        if (enable_logging) {
            if (log_history_length_ms <= 0) {
                MV_LOG_ERROR() << "log_history_length_ms is" << log_history_length_ms
                               << ". This parameter can not be negative.";
                all_ok = false;
            }
            if (log_dump_delay_ms > log_history_length_ms || log_dump_delay_ms < 0) {
                MV_LOG_ERROR() << "log_dump_delay_ms is" << log_dump_delay_ms
                               << "and it has to be >= 0 and < log_history_length_ms (" << log_history_length_ms
                               << ").";
                all_ok = false;
            }
        }

        return all_ok;
    }
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_JET_MONITORING_LOGGER_CONFIG_H
