/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_EVENT_COUNTER_ALGORITHM_H
#define METAVISION_SDK_CV_EVENT_COUNTER_ALGORITHM_H

#include <string>
#include <functional>
#include <fstream>
#include <mutex>
#include <map>
#include <iostream>
#include <iomanip>
#include <list>
#include <type_traits>
#include <opencv2/core.hpp>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/core/utils/detail/iterator_traits.h"

namespace Metavision {

class Event2d;

/// This class counts the number of events happening during a simulation step
class EventCounterAlgorithm {
public:
    /// @brief Builds a new EventCounterAlgorithm object
    /// @param threshold Threshold in average rate for the analog_output overlay in Mev/S
    inline EventCounterAlgorithm(double threshold = 19);

    /// @brief Builds a new EventCounterAlgorithm object
    /// @param output_filename Name of the output file to be generated
    /// @param step_time Time interval (in us) used to compute the peak event rate
    /// @param scaling_time Time interval (in us) used to compute the average event rate
    /// @param enable_log true to save a log file
    /// @param threshold Threshold in average rate for the analog_output overlay in Mev/S
    /// @param compute_variance true to compute the variance
    inline EventCounterAlgorithm(const std::string &output_filename, timestamp step_time, timestamp scaling_time,
                                 bool enable_log = true, double threshold = 19., bool compute_variance = false);

    /// @brief Destructor
    ~EventCounterAlgorithm() {
        stop_log();
    }

    /// @brief Enables or disables data logging
    /// @param state If true, the logging will start, otherwise it will stop
    inline void enable_log(bool state);

    /// @brief Sets the output file
    inline void set_log_destination(const std::string &csv_file_name);

    /// @brief Sets step time
    /// @param step_time_us Time interval (in us) used to compute the peak event rate during the step time
    /// (must be smaller than scaling time). Default is 1 ms.
    inline void set_step_time_us(timestamp step_time_us);

    /// @brief Sets scaling time
    /// @param scaling_time_us Time interval (in us) used to compute the average event rate during the
    /// scaling time. Default is 1000 ms
    inline void set_scaling_time_us(timestamp scaling_time_us);

    /// @brief Counts only the events that have the input polarity.
    ///
    /// All events are counted by default.
    /// Use -1 to disable the filter after setting a value
    /// If the event doesn't have a polarity, the compilation will fail
    /// @param polarity The polarity to count
    inline void polarity_to_count(int polarity);

    /// @brief Prints statistics
    inline void print_stats();

    /// @brief Gets the average rate last computed
    /// @return End time (in seconds) of the last average rate that has been computed (first element of the pair)
    ///         and the average rate (kEv per second) during this interval
    inline std::pair<double, double> get_average_rate();

    /// @brief Gets the peak rate last computed
    /// @return Time (in seconds) of the last peak rate that has been computed (first element of the pair)
    ///         and the average peak rate (kEv per second)
    inline std::pair<double, double> get_peak_rate();

    /// @brief Gets number of events
    /// @return Total number of events counted from the input producer since the beginning
    inline uint64_t get_events_number();

    /// @brief Gets number of events of the polarity set by @ref polarity_to_count
    /// @return Total number of events of the polarity set by @ref polarity_to_count counted from the input producer
    /// since the beginning
    inline uint64_t get_events_number_by_polarity();

    /// @brief Adds a function to be called every time an event rate has been computed
    /// @param cb Function that takes as input the timestamp (in seconds) of the last event the average event rate has
    /// been computed on and the average event rate (kEv per second)
    /// @return Callback id
    inline size_t add_callback_on_scale(const std::function<void(double, double)> &cb);

    /// @brief Removes the function to call every time an event rate has been computed
    /// @param cb_id Callback id
    inline void remove_callback_on_scale(size_t cb_id);

    /// @brief Processes a set of events using iterators
    /// @param it Iterator to the first event
    /// @param it_end Iterator to the past-the-end event
    /// @param ts Timestamp of current timeslice
    /// @note Two implementations are provided, the first one for events derived from Event2d,
    ///       the other one for non Event2d-derived events
    template<typename InputIt>
    inline std::enable_if_t<
        !std::is_base_of<Metavision::Event2d, typename Metavision::iterator_traits<InputIt>::value_type>::value>
        process(InputIt it, InputIt it_end, timestamp ts);

    template<typename InputIt>
    inline std::enable_if_t<
        std::is_base_of<Metavision::Event2d, typename Metavision::iterator_traits<InputIt>::value_type>::value>
        process(InputIt it, InputIt it_end, timestamp ts);

    // Default output interface to generate analog buffers
    inline void process_output(cv::Mat &output);

    // This interface is used in the EventCounterEventRate to generate the output buffer
    inline void process_output_er(cv::Mat &output);

    inline void reset();

private:
    inline void update(timestamp ts);
    inline void update_output(timestamp ts, int buffer_id, bool analog_output_needed);
    inline void start_log();
    inline void stop_log();
    inline void do_scale();
    inline void do_step();
    inline void do_variance_step(const double ev_x, const double ev_y);
    inline void write_stats();

    std::string output_filename_;
    std::ofstream output_csv_file_;
    int polarity_to_count_ = -1;

    timestamp step_time_us_    = 1000;
    timestamp scaling_time_us_ = 1000000;

    bool log_enabled_             = false;
    bool variance_computed_       = false;
    uint64_t step_events_counter_ = 0;

    timestamp origin_       = 0;
    timestamp ts_step_end_  = step_time_us_;
    timestamp ts_scale_end_ = scaling_time_us_;

    uint64_t scaling_events_counter_ = 0;
    uint64_t max_step_counter_       = 0;

    double last_average_rate_kEv_s_ = 0;
    timestamp ts_last_average_rate_ = 0;
    double last_peak_rate_kEv_s_    = 0;
    timestamp ts_last_peak_rate_    = 0;
    double mean_x_                  = 0;
    double mean_y_                  = 0;
    double M2_x_                    = 0;
    double M2_y_                    = 0;
    double variance_                = 0;

    uint64_t n_total_events_          = 0;
    uint64_t n_total_events_polarity_ = 0;

    // Keep trace if the next interval time steps changed
    // (you cannot change the time steps while computing an interval)
    bool step_time_changed_        = false;
    bool scaling_time_changed_     = false;
    timestamp new_step_time_us_    = 1000;
    timestamp new_scaling_time_us_ = 1000000;

    std::mutex mutex_;
    std::map<size_t, std::function<void(double, double)>> cbs_map_;
    // visualization
    double threshold_;
    cv::Mat analog_buffer_ = cv::Mat::zeros(1, 1, CV_64FC1);
};

EventCounterAlgorithm::EventCounterAlgorithm(double threshold) {
    threshold_ = 1000. * threshold;
}

EventCounterAlgorithm::EventCounterAlgorithm(const std::string &output_filename, timestamp step_time,
                                             timestamp scaling_time, bool enable_log, double threshold,
                                             bool compute_variance) :
    output_filename_(output_filename), log_enabled_(enable_log), variance_computed_(compute_variance) {
    scaling_time_us_ = scaling_time;
    step_time_us_    = step_time;
    threshold_       = 1000. * threshold;

    if (scaling_time % step_time != 0) {
        step_time_us_ = scaling_time_us_ / (scaling_time_us_ / step_time);
        MV_SDK_LOG_WARNING() << "Step time has to be a divisor of the scaling time. Setting step time to"
                             << step_time_us_;
    }

    if (log_enabled_) {
        output_csv_file_.open(output_filename_.c_str());

        if (output_csv_file_.fail()) {
            MV_SDK_LOG_WARNING() << Log::no_space << "Unable to open file " << output_filename_
                                 << ": no output will be generated.";
            log_enabled_ = false;
        }
    }

    ts_step_end_  = origin_ + step_time_us_;
    ts_scale_end_ = origin_ + scaling_time_us_;
}

// process() method for Event2d-derived events
template<typename InputIt>
std::enable_if_t<std::is_base_of<Metavision::Event2d, typename Metavision::iterator_traits<InputIt>::value_type>::value>
    EventCounterAlgorithm::process(InputIt it, InputIt it_end, timestamp ts) {
    const size_t n_events = std::distance(it, it_end);
    n_total_events_ += n_events;

    for (; it != it_end; ++it) {
        if (polarity_to_count_ != -1) {
            if (it->p != polarity_to_count_) {
                continue;
            }
        }
        ++n_total_events_polarity_;
        const timestamp ev_ts = it->t; // current timestamp
        while (ev_ts > ts_step_end_) {
            do_step();

            if (ts_step_end_ > ts_scale_end_) {
                do_scale(); // end the scaling time interval
            }
        }

        ++step_events_counter_;
        if (variance_computed_) {
            const double ev_x = it->x; // current x
            const double ev_y = it->y; // current y
            do_variance_step(ev_x, ev_y);
        }
    }
    while (ts > ts_step_end_) {
        do_step();

        if (ts_step_end_ > ts_scale_end_) {
            do_scale(); // end the scaling time interval
        }
    }
}

// process() method for non Event2d-derived events: not considering polarities and 2D coordinates
template<typename InputIt>
std::enable_if_t<
    !std::is_base_of<Metavision::Event2d, typename Metavision::iterator_traits<InputIt>::value_type>::value>
    EventCounterAlgorithm::process(InputIt it, InputIt it_end, timestamp ts) {
    const size_t n_events = std::distance(it, it_end);
    n_total_events_ += n_events;

    for (; it != it_end; ++it) {
        const timestamp ev_ts = it->t; // current timestamp
        while (ev_ts > ts_step_end_) {
            do_step();

            if (ts_step_end_ > ts_scale_end_) {
                do_scale(); // end the scaling time interval
            }
        }

        ++step_events_counter_;
    }
    while (ts > ts_step_end_) {
        do_step();

        if (ts_step_end_ > ts_scale_end_) {
            do_scale(); // end the scaling time interval
        }
    }
}

void EventCounterAlgorithm::process_output_er(cv::Mat &output) {
    std::pair<double, double> pair  = get_average_rate();
    analog_buffer_.at<double>(0, 0) = pair.second;
    analog_buffer_.copyTo(output);
}

void EventCounterAlgorithm::reset() {
    scaling_events_counter_  = 0;
    max_step_counter_        = 0;
    n_total_events_          = 0;
    n_total_events_polarity_ = 0;
    last_average_rate_kEv_s_ = 0;
    ts_last_average_rate_    = 0;
    last_peak_rate_kEv_s_    = 0;
    mean_x_                  = 0;
    mean_y_                  = 0;
    M2_x_                    = 0;
    M2_y_                    = 0;
    variance_                = 0;
}

void EventCounterAlgorithm::process_output(cv::Mat &output) {
    if (last_average_rate_kEv_s_ > threshold_) {
        analog_buffer_.setTo(cv::Scalar(1.));
    } else {
        analog_buffer_.setTo(cv::Scalar(0.));
    }
    analog_buffer_.copyTo(output);
}

void EventCounterAlgorithm::start_log() {
    if (log_enabled_)
        return;

    // output_csv_file_ is already closed, no need to close it (otherwise there is a bug...)
    if (output_filename_ == "") {
        log_enabled_ = false;
        MV_SDK_LOG_WARNING() << "Cannot log because the output filename is not set";
        return;
    }
    output_csv_file_.open(output_filename_.c_str(), std::ios::trunc);

    if (!output_csv_file_.is_open()) {
        log_enabled_ = false;
        MV_SDK_LOG_WARNING() << "Could not open file at" << output_filename_;
    } else {
        log_enabled_ = true;
    }
}

void EventCounterAlgorithm::stop_log() {
    if (!log_enabled_)
        return;

    if (output_csv_file_.is_open()) {
        output_csv_file_.close();
    }

    log_enabled_ = false;
}

void EventCounterAlgorithm::do_step() {
    // compute peak: if max then peak else not a peak
    if (step_events_counter_ > max_step_counter_) {
        max_step_counter_     = step_events_counter_;
        ts_last_peak_rate_    = ts_step_end_;
        last_peak_rate_kEv_s_ = (max_step_counter_ / 1000.) / (step_time_us_ / 1000000.);
    }

    scaling_events_counter_ += step_events_counter_;

    step_events_counter_ = 0;
    ts_step_end_ += step_time_us_;
}

void EventCounterAlgorithm::do_variance_step(const double ev_x, const double ev_y) {
    // compute variance online using Welford algorithm
    const double delta_x = ev_x - mean_x_;
    const double delta_y = ev_y - mean_y_;
    mean_x_ += delta_x / step_events_counter_;
    mean_y_ += delta_y / step_events_counter_;
    const double delta2_x = ev_x - mean_x_;
    const double delta2_y = ev_y - mean_y_;
    M2_x_ += delta_x * delta2_x;
    M2_y_ += delta_y * delta2_y;
}

void EventCounterAlgorithm::do_scale() {
    // the scaling time interval ended
    last_average_rate_kEv_s_ = (scaling_events_counter_ / 1000.) / (scaling_time_us_ / 1000000.);
    ts_last_average_rate_    = ts_scale_end_;
    if (variance_computed_) {
        variance_ = (M2_x_ + M2_y_) / scaling_events_counter_;
        mean_x_   = 0;
        mean_y_   = 0;
        M2_x_     = 0;
        M2_y_     = 0;
    }

    max_step_counter_       = 0;
    scaling_events_counter_ = 0;

    if (scaling_time_changed_) {
        scaling_time_us_      = new_scaling_time_us_;
        scaling_time_changed_ = false;
    }

    if (step_time_changed_) {
        step_time_us_      = new_step_time_us_;
        ts_step_end_       = ts_scale_end_ + step_time_us_;
        step_time_changed_ = false;
    }

    ts_scale_end_ += scaling_time_us_;

    if (log_enabled_) {
        write_stats();
    }

    // Copy the map before calling the functions (to avoid infinite loops)
    std::list<std::function<void(double, double)>> cbs;
    {
        // multi-threading protection from map modifications
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto &&p : cbs_map_) {
            cbs.push_back(p.second);
        }
    }

    // Call the callbacks
    double ts_last_s = ts_last_average_rate_ / 1000000.;
    for (auto &&cb : cbs)
        cb(ts_last_s, last_average_rate_kEv_s_);
}

void EventCounterAlgorithm::write_stats() {
    if (variance_computed_) {
        output_csv_file_ << std::setprecision(6) << std::fixed << static_cast<double>(ts_last_average_rate_) / 1000000.
                         << "," << last_average_rate_kEv_s_ << "," << static_cast<double>(ts_last_peak_rate_) / 1000000.
                         << "," << last_peak_rate_kEv_s_ << "," << variance_ << std::endl;
    } else {
        output_csv_file_ << std::setprecision(6) << std::fixed << static_cast<double>(ts_last_average_rate_) / 1000000.
                         << "," << last_average_rate_kEv_s_ << "," << static_cast<double>(ts_last_peak_rate_) / 1000000.
                         << "," << last_peak_rate_kEv_s_ << std::endl;
    }
}

void EventCounterAlgorithm::print_stats() {
    auto log = MV_SDK_LOG_INFO() << Log::no_space << std::setprecision(6);
    if (variance_computed_) {
        if (polarity_to_count_ != -1) {
            log << "Polarity -> \t" << polarity_to_count_ << " | \t";
        }
        log << static_cast<double>(ts_last_average_rate_) / 1000000. << "s -> \t" << last_average_rate_kEv_s_
            << " kEv/s average | \t"
            << "Peak -> " << last_peak_rate_kEv_s_ << " kEv/s | \t"
            << "Variance -> " << variance_;
    } else {
        if (polarity_to_count_ != -1) {
            log << "Polarity -> \t" << polarity_to_count_ << " | \t";
        }
        log << static_cast<double>(ts_last_average_rate_) / 1000000. << "s -> \t" << last_average_rate_kEv_s_
            << " kEv/s average | \t"
            << "Peak -> " << last_peak_rate_kEv_s_ << " kEv/s";
    }
}

void EventCounterAlgorithm::set_step_time_us(timestamp step_time) {
    if (step_time > scaling_time_us_) {
        MV_SDK_LOG_ERROR() << "Step time must be lower or equal than scaling time. The step time will not be modified.";
        return;
    }

    new_step_time_us_ = step_time;

    if (scaling_time_us_ % new_step_time_us_ != 0) {
        new_scaling_time_us_ = step_time * (scaling_time_us_ / new_step_time_us_);

        MV_SDK_LOG_WARNING() << "Scaling time has to be a multiple of the step time. Setting scaling time to"
                             << new_scaling_time_us_;
        scaling_time_changed_ = true;
    }

    step_time_changed_ = true;
}

void EventCounterAlgorithm::set_scaling_time_us(timestamp scaling_time) {
    if (step_time_us_ > scaling_time) {
        MV_SDK_LOG_WARNING() << "Scaling time must be greater than step time. The scaling time will not be modified";
        return;
    }

    new_scaling_time_us_ = scaling_time;

    if (new_scaling_time_us_ % step_time_us_ != 0) {
        new_step_time_us_ = new_scaling_time_us_ / (new_scaling_time_us_ / step_time_us_);

        MV_SDK_LOG_WARNING() << "Step time has to be a divisor of the scaling time. Setting step time to"
                             << Log::no_space << new_step_time_us_ << "us";
        step_time_changed_ = true;
    }

    scaling_time_changed_ = true;
}

void EventCounterAlgorithm::enable_log(bool state) {
    state ? start_log() : stop_log();
}

void EventCounterAlgorithm::set_log_destination(const std::string &csv_file_name) {
    if (csv_file_name.size() == 0) {
        MV_SDK_LOG_ERROR() << "Invalid filename specified.";
        return;
    }

    if (log_enabled_ && csv_file_name == output_filename_) {
        MV_SDK_LOG_WARNING() << Log::no_space << "You already are writing data in file " << output_filename_
                             << ": file will be overwritten.";
    }

    output_filename_ = csv_file_name;
    if (log_enabled_) {
        stop_log();
        start_log();
    }
}

std::pair<double, double> EventCounterAlgorithm::get_average_rate() {
    return std::make_pair(static_cast<double>(ts_last_average_rate_) / 1e6, last_average_rate_kEv_s_);
}

std::pair<double, double> EventCounterAlgorithm::get_peak_rate() {
    return std::make_pair(static_cast<double>(ts_last_peak_rate_) / 1e6, last_peak_rate_kEv_s_);
}

uint64_t EventCounterAlgorithm::get_events_number() {
    return n_total_events_;
}
uint64_t EventCounterAlgorithm::get_events_number_by_polarity() {
    return n_total_events_polarity_;
}

size_t EventCounterAlgorithm::add_callback_on_scale(const std::function<void(double, double)> &cb) {
    // multi-threading protection from map modifications
    std::lock_guard<std::mutex> lock(mutex_);
    size_t id    = (cbs_map_.empty() ? 0 : cbs_map_.rbegin()->first + 1);
    cbs_map_[id] = cb;
    return id;
}

void EventCounterAlgorithm::remove_callback_on_scale(size_t cb_id) {
    // multi-threading protection from map modifications
    std::lock_guard<std::mutex> lock(mutex_);
    cbs_map_.erase(cb_id);
}

void EventCounterAlgorithm::polarity_to_count(int polarity) {
    polarity_to_count_ = polarity;
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_EVENT_COUNTER_ALGORITHM_H
