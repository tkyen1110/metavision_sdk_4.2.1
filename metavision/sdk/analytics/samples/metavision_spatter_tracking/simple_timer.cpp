/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include "simple_timer.h"
#include <metavision/sdk/base/utils/sdk_log.h>

SimpleTimer::~SimpleTimer() {
    if (is_timing_) {
        stop_timer_   = std::chrono::steady_clock::now();
        save_time_to_ = last_ts_;
    }

    std::uint64_t time = std::chrono::duration_cast<std::chrono::microseconds>(stop_timer_ - start_timer_).count();
    MV_SDK_LOG_INFO() << "Time spent to process" << (double)(save_time_to_ - save_time_from_) / 1000
                      << "ms of data is:" << (double)(time) / 1000 << "ms";
}

void SimpleTimer::update_timer(const Metavision::timestamp ts) {
    if (is_timer_ended_)
        return;
    if (!is_timing_) {
        if (ts > save_time_from_) {
            start_timer_    = std::chrono::steady_clock::now();
            is_timing_      = true;
            save_time_from_ = ts;
        }
    } else {
        if (ts > save_time_to_) {
            stop_timer_     = std::chrono::steady_clock::now();
            is_timing_      = false;
            is_timer_ended_ = true;
        }
    }
    last_ts_ = ts;
}

void SimpleTimer::set_time_range(const Metavision::timestamp &time_from, const Metavision::timestamp &time_to) {
    if (time_from > 0)
        save_time_from_ = time_from;
    if (time_to > 0)
        save_time_to_ = time_to;
}
