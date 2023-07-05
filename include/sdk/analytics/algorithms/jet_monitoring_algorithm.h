/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_JET_MONITORING_ALGORITHM_H
#define METAVISION_SDK_ANALYTICS_JET_MONITORING_ALGORITHM_H

#include <limits>
#include <functional>
#include <memory>
#include <boost/circular_buffer.hpp>
#include <opencv2/core/types.hpp>

#include "metavision/sdk/core/algorithms/async_algorithm.h"
#include "metavision/sdk/analytics/configs/jet_monitoring_configs.h"
#include "metavision/sdk/analytics/utils/filtered_bool.h"
#include "metavision/sdk/analytics/utils/jet_monitoring_alarms.h"
#include "metavision/sdk/analytics/utils/rectangle_roi_utils.h"

namespace Metavision {

struct JetMonitoringSliceData;

/// @brief Class that detects, counts, and timestamps jets that are being dispensed.
///
/// The algorithm starts by splitting the Region Of Interest (ROI) provided by the user into three parts.
/// On the one hand, the central ROI is used to detect jets by identifying peaks in the event-rate.
/// On the other hand, the two surrounding ROIs are used to analyze the background activity.
///
/// Jet Monitoring results are provided through callbacks to which the user can subscribe.
/// The first two provide the Jet Monitoring results:
///   - JetCallback  : when a jet is detected
///   - AlarmCallback: when an alarm is raised
///
/// While the other two provide contextual information on the time slice that has just been processed:
///   - SliceCallback: detailed information about the time slice (See @ref JetMonitoringSliceData)
///   - AsyncCallback: end-timestamp and number of events of the time slice
class JetMonitoringAlgorithm : public AsyncAlgorithm<JetMonitoringAlgorithm> {
public:
    friend class AsyncAlgorithm<JetMonitoringAlgorithm>;

    /// @brief Enum listing ROIs for counting and background activity
    enum ROI {
        DETECTION = 0, // Central ROI, to count events to look for a jet
        BG_NOISE_1,    // 1,2: ROIs located at both sides, to look for background activity
        BG_NOISE_2,    //
        TOTAL,         // Total number of ROIs
    };

    /// @brief Type of callback called when a jet is detected (Main interface)
    using JetCallback = std::function<void(const EventJet &)>;
    /// @brief Type of callback called when an alarm is raised (Main interface)
    using AlarmCallback = JetMonitoringAlarms::AlarmCallback;

    /// @brief Type of callback called at the end of each time slice processing, with detailed information about the
    /// internal state. It is mainly intended for diagnostic/debugging.
    using SliceCallback = std::function<void(const JetMonitoringSliceData &)>;

    /// @brief Type of callback called at the end of each time slice processing, with the timestamp and the number of
    /// events.
    ///
    /// In case the @ref SliceCallback has also been specified, the AsyncCallback will be called last
    using AsyncCallback = std::function<void(const timestamp, const size_t)>;

    /// @brief Constructor
    /// @param algo_config Jet monitoring parameters
    /// @param alarm_config Jet monitoring alarm parameters
    JetMonitoringAlgorithm(const JetMonitoringAlgorithmConfig &algo_config,
                           const JetMonitoringAlarmConfig &alarm_config = JetMonitoringAlarmConfig());

    /// @brief Resets internal state
    void reset_state();

    /// @brief Sets the callback that is called when a jet is detected
    /// @param cb Callback processing a const reference of @ref EventJet
    void set_on_jet_callback(const JetCallback &cb);

    /// @brief Sets the callback that is called when an alarm is raised
    /// @param cb Callback processing a const reference of @ref EventJetAlarm
    void set_on_alarm_callback(const AlarmCallback &cb);

    /// @brief Sets the callback that is called at the end of each slice to provide JetMonitoring-related data
    /// @param cb Callback processing a const reference of @ref JetMonitoringSliceData
    void set_on_slice_callback(const SliceCallback &cb);

    /// @brief Sets the callback that is called at the end of each slice to provide AsyncAlgorithm-related data
    /// @param cb Callback processing the time slice duration and the number of events processsed during time slice
    void set_on_async_callback(const AsyncCallback &cb);

private:
    /// @brief Callback called on each jet
    JetCallback jet_cb_;

    /// @brief Callback called on each time slice
    SliceCallback slice_cb_;

    /// @brief Callback called on each time slice
    AsyncCallback async_cb_;

    /// @brief Slice duration in us
    int slice_duration_us_ = 0;

    /// @brief How many slices to accumulate for the counters
    int n_slices_to_accumulate_ = 0;

    std::array<RectangleRoi, ROI::TOTAL> rois_; ///< ROIS for counting and background activity

    /// @{
    /// @brief Thresholds for detection/timing
    int th_up_   = 0;
    int th_down_ = 0;
    FilteredBool is_er_above_th_up_;
    FilteredBool is_er_above_th_down_;
    ///@}

    /// @brief Ts when the ER goes above the UP threshold for the first time for a given jet
    timestamp first_rising_edge_ts_{0};

    /// @brief A jet has been detected and it is still present
    bool jet_is_present_ = false;

    long jet_count_            = 0; ///< Number of jets counted
    timestamp previous_jet_ts_ = 0; ///< Timestamp of the last time a jet was detected
    timestamp previous_jet_dt_ = 0; ///< Timestamp difference between the last two detected jets

    using Counters = std::array<std::array<int, 2>, ROI::TOTAL>;

    /// @brief Value of the counters for the slice being processed
    Counters current_slice_evt_counters_;

    /// @brief Keeps a history of the counters for the last N+1 slices
    ///
    /// That way, we can compute the average in the last N slices in constant time, by taking the running counters
    /// (avg_evt_counters_) plus the most recent element in this buffer minus the oldest one
    boost::circular_buffer<Counters> evt_counter_history_;

    /// @brief Last updated value of the counters, averaged over the last n_slices_to_accumulate_ slices
    Counters avg_evt_counters_;

    /// @brief Algo for alarm generation.
    std::unique_ptr<JetMonitoringAlarms> alarms_{nullptr};

private:
    /// @brief Processes events for the current time slice (the slice might not be yet complete)
    template<typename InputIt>
    void process_online(const InputIt ev_begin, const InputIt ev_end);

    /// @brief Processes a time-slice of slice_duration_us_
    void process_async(const timestamp processing_ts, const size_t n_processed_events);

    /// @brief Updates counter history and accumulated values
    void update_history_and_accumulate();

    /// @brief Internal processing function that detects a jet using the current history
    void detect_jet(const timestamp processing_ts, const size_t n_processed_events);
};

template<typename InputIt>
void JetMonitoringAlgorithm::process_online(const InputIt ev_begin, const InputIt ev_end) {
    // Here, we count the events in the different ROIs in tmp_evt_counters.
    // tmp_evt_counters_ should be reset when the slice is processed in process_async()
    accumulate_events_rectangle_roi(ev_begin, ev_end, rois_.cbegin(), rois_.cend(),
                                    current_slice_evt_counters_.begin());
}

/// @brief Structure that holds the data obtained by processing a time slice with @ref JetMonitoringAlgorithm
struct JetMonitoringSliceData {
    /// @brief Timestamp at the end of the slice
    timestamp slice_end_ts = 0;

    /// @brief Total number of events in the slice (to log the camera ER.)
    long n_evts = 0;

    /// @brief Number of positive events for all ROIs (not averaged, in # of events per slice)
    float roi_pos_ev_count[JetMonitoringAlgorithm::ROI::TOTAL];

    /// @brief Number of negative events for all ROIs (not averaged, in # of events per slice)
    float roi_neg_ev_count[JetMonitoringAlgorithm::ROI::TOTAL];

    /// @brief Number of positive event rate for all ROIs (averaged over the accumulation time, in kEv/s)
    float roi_avg_er_pos_kevps[JetMonitoringAlgorithm::ROI::TOTAL];

    /// @brief Number of negative event rate for all ROIs (averaged over the accumulation time, in kEv/s)
    float roi_avg_er_neg_kevps[JetMonitoringAlgorithm::ROI::TOTAL];

    /// @brief Final (averaged) event rate, in kEv/s, used for detection
    float avg_er_pos_kevps_detection = 0;

    // Internal variables for jet detection
    bool jet_is_present        = false;
    bool is_er_above_th_up     = false;
    bool is_er_above_th_up_raw = false;

    // Algo results
    long jet_count              = 0; // Number of jets counted so far.
    timestamp last_jet_dt       = 0; // Timestamp difference between the last two detected jets.
    timestamp ts_since_last_jet = 0; // Timestamp elapsed since the last detected jet.
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_JET_MONITORING_ALGORITHM_H