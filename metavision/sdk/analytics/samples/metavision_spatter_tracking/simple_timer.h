/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_SIMPLE_TIMER_H
#define METAVISION_SDK_ANALYTICS_SIMPLE_TIMER_H

#include <iostream>
#include <atomic>
#include <chrono>

#include <metavision/sdk/base/utils/timestamp.h>

/// @brief Class measuring the processing time in a specific data time range
class SimpleTimer {
public:
    SimpleTimer() = default;

    /// @brief Prints timing on destruction
    ~SimpleTimer();

    /// @brief Updates the timer status using the current timestamp
    /// @param ts Current timestamp
    void update_timer(const Metavision::timestamp ts);

    /// @brief Specifies the data time range for which the processing time will be measured
    /// @param time_from Beginning of the time range
    /// @param time_to End of the time range
    void set_time_range(const Metavision::timestamp &time_from, const Metavision::timestamp &time_to);

private:
    // Time from the timestamp
    Metavision::timestamp save_time_from_ = 0;
    // Time up to the timestamp
    Metavision::timestamp save_time_to_ = std::numeric_limits<Metavision::timestamp>::max();

    // Timers
    std::chrono::time_point<std::chrono::steady_clock> start_timer_;
    std::chrono::time_point<std::chrono::steady_clock> stop_timer_;

    std::atomic_bool is_timing_{false};
    bool is_timer_ended_ = false;

    Metavision::timestamp last_ts_ = 0;
};

#endif // METAVISION_SDK_ANALYTICS_SIMPLE_TIMER_H
