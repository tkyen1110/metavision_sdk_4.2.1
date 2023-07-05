/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_CSV_LOGGER_H
#define METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_CSV_LOGGER_H

#include <fstream>
#include <string>

#include "metavision/sdk/analytics/events/event_spatter_cluster.h"
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Class that logs the output of @ref SpatterTrackerAlgorithm to csv format
class SpatterTrackerCsvLogger {
public:
    /// @brief Constructor
    /// @param file_name Output file name
    SpatterTrackerCsvLogger(std::string file_name) {
        output_file_.open(file_name + ".csv");
        if (output_file_.fail()) {
            throw std::runtime_error(
                "Could not open file '" + file_name +
                " to record. Make sure it is a valid filename and that you have permissions to write it.");
        }
    }

    ~SpatterTrackerCsvLogger() {
        output_file_.close();
    }

    /// @brief Writes information about the trackers to file
    /// @param ts Timestamp
    /// @param trackers Trackers to write
    void log_output(const Metavision::timestamp ts, const std::vector<EventSpatterCluster> &trackers) {
        // No need to check if file is open (done in the ctor)
        for (const auto &ev : trackers) {
            output_file_ << ts << ", " << ev.id << ", " << ev.x << ", " << ev.y << ", " << ev.width << ", " << ev.height
                         << std::endl;
        }
    }

private:
    // File to write the log
    std::ofstream output_file_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_CSV_LOGGER_H
