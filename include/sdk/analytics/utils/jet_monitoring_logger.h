/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_JET_MONITORING_LOGGER_H
#define METAVISION_SDK_ANALYTICS_JET_MONITORING_LOGGER_H

#include <string>
#include <fstream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <queue>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/core/algorithms/stream_logger_algorithm.h"
#include "metavision/sdk/analytics/algorithms/jet_monitoring_algorithm.h"
#include "metavision/sdk/analytics/configs/jet_monitoring_configs.h"

namespace bfs = boost::filesystem;

namespace Metavision {

/// @brief Class maintaining circular buffers with data of interest concerning jet monitoring, and dumping it to
/// a set of files each time a log trigger arrives:
///   - events_td.dat:        CD events
///   - monitoring_out.csv:   Algorithm output
///   - algo_parameters.json: Algorithm parameters
///
/// The intended use is:
///  - Call process_events() to register events
///  - Call log() to register JetMonitoringSliceData
///  - (if needed) call  schedule_dump(), this will dump data after the specified delay
///  - Call erase data to discard unnecessary data
class JetMonitoringLogger {
public:
    /// @brief Constructor: initializes the logger
    /// @param sensor_size Sensor size (width, height) in pixels, needed to log events
    /// @param roi_camera Camera ROI
    /// @param algo_config Jet Monitoring Algorithm configuration parameters
    /// @param alarm_config Jet Monitoring Alarm configuration parameters
    /// @param logger_config Jet Monitoring Logger configuration parameters
    JetMonitoringLogger(cv::Size sensor_size, const cv::Rect &roi_camera,
                        const JetMonitoringAlgorithmConfig &algo_config, const JetMonitoringAlarmConfig &alarm_config,
                        const JetMonitoringLoggerConfig &logger_config);

    /// @brief Destructor
    ~JetMonitoringLogger();

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end);

    /// @brief Logs data to the buffer and erases old buffers that won't be used for log dumps
    /// @param history_item Item to be logged in the buffer
    void log(const JetMonitoringSliceData &history_item);

    /// @brief Processes alarm (dumps data)
    /// @param alarm Alarm information from the Jet Monitoring algorithm
    void process_alarm(const EventJetAlarm &alarm);

    /// @brief Schedules a log data dump, after the delay specified in the constructor by dump_delay
    /// @note If several dumps are requested for the same time slice, only the last one will be kept
    /// @param ts Timestamp of the dump
    /// @param trigger_description Description of what triggered the dump, this will be appended to the
    /// dump directory name
    void schedule_dump(const timestamp ts, const std::string &trigger_description);

private:
    /// @brief Struct storing the begin and end timestamps of a time slice
    struct TimeSlice {
        TimeSlice() = default;

        timestamp ts_begin;
        timestamp ts_end;
    };

    /// @brief Processes dump requests
    void process_dump_requests();

    /// @brief Dumps data related to a new jet
    void dump_new_jet_data(const timestamp trigger_ts);

    /// @brief Dumps the buffers to a new sub directory
    void dump_buffers(timestamp trigger_ts, const std::string &suffix);

    /// @brief Dumps the algorithm output buffer to a CSV file
    void dump_csv_data(const timestamp ts, const bfs::path &file_path);

    /// @brief Dumps the event buffer to a DAT file
    void dump_event_data(const timestamp ts, const bfs::path &file_path);

    /// @brief Dumps the application options to a CSV file
    void dump_app_options(const bfs::path &file_path);

    /// @brief Erases data that is not needed for dump, i.e. with a timestamp older than the beginning of the log dump
    /// @warning It does not mean that each log older than @p ts will be erased
    /// @param ts Current timestamp
    void erase_buffers(const timestamp ts);

    /// @brief Gets dump time slice corresponding to the given trigger timestamp
    /// @param trigger_ts Trigger timestamp
    /// @return Begin and end timestamps of the triggered time slice, which is of length dump_length_ and ends at
    /// @p trigger_ts + dump_delay_
    TimeSlice get_dump_time_slice(const timestamp trigger_ts) const;

    // Other auxiliary functions

    /// @brief Creates a directory if it doesn't exist
    /// @throw std::runtime_error if it cannot not create the directory
    static void make_directory(const bfs::path &dir_name);

    // Data buffers. Index [0] is the most recent.

    /// @brief Buffer of history items
    std::vector<JetMonitoringSliceData> history_buffer_;

    /// @brief Buffer of events
    std::vector<EventCD> evt_buffer_;

    /// @brief Dump string that indicates that we should erase data
    static const std::string ERASE_DATA_STR;

    /// @brief Dump string that indicates that a new jet was detected
    static const std::string NEW_JET_STR;

    /// @brief Buffer that maintains the dump requests. A string equal to ERASE_DATA_STR indicates that we should
    // erase data. Else, it indicates the suffix for the dump directoy name.
    std::queue<std::pair<timestamp, std::string>> dump_requests_;

    /// @brief Timestamp of the last received event
    timestamp ts_last_ev_;

    /// @brief Timestamp of the last received history item
    timestamp ts_last_history_;

    /// @brief Length of the data to dump
    timestamp dump_length_;

    /// @brief Length of the data to dump after the ts of the dump
    timestamp dump_delay_ = 0;

    /// @brief Base output directory. Each dump will generate a sub directory in it
    bfs::path output_base_path_;

    /// @brief Prefix for each output directory
    std::string output_prefix_;

    /// @brief Sensor size, in pixels, needed to dump .dat events
    cv::Size sensor_size_;

    /// @brief Logs the event rate for each jet
    bool log_jets_evt_rate_ = false;

    /// @brief Vector of event rate for each jet
    std::vector<std::vector<int>> jet_roi_counts_;

    /// @brief Create a video of an average jet
    bool log_jet_video_ = false;

    /// @brief List of accumulated events on the current ROI
    std::vector<cv::Mat1i> avg_jet_images_;

    /// @brief Number of frames generated for the jet video
    static constexpr int N_SLICE = 50;

    /// @brief Number of counted jets
    int jet_count_ = 0;

    /// @brief Jet is present
    bool jet_is_present_ = false;

    /// @brief Camera roi. Used for the avg_jet_images
    cv::Rect roi_camera_;

    JetMonitoringAlgorithmConfig algo_config_;
    JetMonitoringAlarmConfig alarm_config_;
    JetMonitoringLoggerConfig logger_config_;
};

template<typename InputIt>
void JetMonitoringLogger::process_events(InputIt it_begin, InputIt it_end) {
    if (it_begin != it_end) {
        evt_buffer_.insert(evt_buffer_.cend(), it_begin, it_end);
        ts_last_ev_ = std::prev(it_end)->t;

        process_dump_requests();
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_JET_MONITORING_LOGGER_H
