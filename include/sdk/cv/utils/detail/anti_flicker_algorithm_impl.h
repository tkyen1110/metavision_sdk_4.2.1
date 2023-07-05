/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_ANTI_FLICKER_ALGORITHM_IMPL_H
#define METAVISION_SDK_CV_ANTI_FLICKER_ALGORITHM_IMPL_H

namespace Metavision {

template<class InputIt, class OutputIt>
OutputIt AntiFlickerAlgorithm::process_events(InputIt it_begin, InputIt it_end, OutputIt inserter) {
    for (auto it = it_begin; it != it_end; ++it)
        if (do_keep_event(*it))
            *inserter++ = Event2d(it->x, it->y, it->p, it->t);
    return inserter;
}

template<class InputEvent>
bool AntiFlickerAlgorithm::do_keep_event(InputEvent event) {
    const unsigned int ev_base_idx = event.y * width_ + event.x;
    PixData &pix_state             = state_[ev_base_idx];

    // Initialize pix_state for this polarity
    if (pix_state.burst_first_ts[event.p] == timestamp(-1)) {
        pix_state.burst_first_ts[event.p] = event.t;
        pix_state.last_pol                = event.p;
        return true;
    }

    // Did the polarity change?
    if (event.p != pix_state.last_pol) {
        const period_precision value = event.t - pix_state.burst_first_ts[event.p];
        const period_precision diff =
            pix_state.last_meas > value ? pix_state.last_meas - value : value - pix_state.last_meas;

        // Did the period change?
        if (diff < diff_thresh_) {
            if (pix_state.cur_count < filter_length_) {
                ++pix_state.cur_count;
            }
        } else {
            pix_state.cur_count = 1;
            pix_state.last_meas = value;
        }

        // We do a convex combination of past values and the new value to denoise
        // the period estimation.
        pix_state.last_meas = 0.95f * pix_state.last_meas + 0.05f * value;

        // Output an event if inside the period range
        pix_state.is_period_valid = pix_state.cur_count >= filter_length_;

        // Start counting next burst
        pix_state.last_pol                = event.p;
        pix_state.burst_first_ts[event.p] = event.t;
    }
    // If the polarity hasn't changed, we consider it still has the same period

    if (!pix_state.is_period_valid)
        return true;

    // Return true if the period is outside the flickering interval
    return (pix_state.last_meas < min_period_ || max_period_ < pix_state.last_meas);
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_ANTI_FLICKER_ALGORITHM_IMPL_H
