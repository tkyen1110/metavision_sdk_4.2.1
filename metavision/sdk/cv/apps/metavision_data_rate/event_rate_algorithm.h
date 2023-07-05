/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_EVENT_RATE_ALGORITHM_H
#define METAVISION_EVENT_RATE_ALGORITHM_H

#include <memory>
#include <sstream>
#include <algorithm>
#include <metavision/sdk/base/utils/timestamp.h>
#include "event_rate_struct.h"

namespace Metavision {

class EventRateAlgorithm {
public:
    EventRateAlgorithm(double min_threshold){
        min_threshold_=min_threshold;
    };

    template<typename InputIt, typename OutputIt>
    OutputIt process_events(InputIt first, InputIt last, OutputIt d_first);

private:
    // Time at which we started to count events
    Metavision::timestamp event_counter_starttime_ = 0.0;

    // Counted events
    int event_counter_ = 0;

    double min_threshold_ = 0;

    // Time during which the rate if counted (in us)
    double timestep_ = 1000.0;
};

template<typename InputIt, typename OutputIt>
OutputIt EventRateAlgorithm::process_events(InputIt first, InputIt last, OutputIt d_first) {
    if (first == last)
        return d_first;

    // initialize the timer
    if (event_counter_starttime_ == 0.0)
        event_counter_starttime_ = first->t;

    event_counter_ += (last - first);

    if (std::prev(last)->t - event_counter_starttime_ > timestep_) {
        // compute the event rate and convert it to kEv/s
        double event_rate_ =
            static_cast<double>(event_counter_) / (std::prev(last)->t - event_counter_starttime_) * 1000.0;
        if (event_rate_ < min_threshold_) {
            event_rate_ = 0;
        }

        // reset
        event_counter_starttime_ = first->t;
        event_counter_           = std::distance(first, last);

        EventRateStruct s_er;
        s_er.rate = static_cast<int>(event_rate_);
        s_er.t    = std::prev(last)->t;
        *d_first  = s_er;
        ++d_first;
    }

    return d_first;
}

} // namespace Metavision

#endif // METAVISION_EVENT_RATE_ALGORITHM_H
