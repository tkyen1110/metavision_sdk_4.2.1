/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_DETAIL_CD_PROCESSING_EVENT_CUBE_H
#define METAVISION_SDK_ML_DETAIL_CD_PROCESSING_EVENT_CUBE_H

#include "metavision/sdk/ml/algorithms/cd_processing_algorithm.h"

#include <cmath>

namespace Metavision {

/// @brief Class used to compute the Event_Cube CD processing
class CDProcessingEventCube : public CDProcessing {
public:
    CDProcessingEventCube() = default;

    /// @brief Constructor
    /// @param delta_t Delta time used to accumulate events inside the frame
    /// @param network_input_width Neural network input frame's width
    /// @param network_input_height Neural network input frame's height
    /// @param num_utbins Number of micro temporal bins
    /// @param split_polarity Process positive and negative events into separate channels
    /// @param max_incr_per_pixel Maximum number of increments per pixel. This is used to normalize the contribution of
    /// each event
    /// @param clip_value_after_normalization Clipping value to apply after normalization (typically: 1.)
    /// @param event_input_width Sensor's width
    /// @param event_input_height Sensor's height
    CDProcessingEventCube(timestamp delta_t, int network_input_width, int network_input_height, int num_utbins,
                          bool split_polarity, float max_incr_per_pixel, float clip_value_after_normalization = 0.f,
                          int event_input_width = 0, int event_input_height = 0) :
        CDProcessing(delta_t, network_input_width, network_input_height, event_input_width, event_input_height, true),
        normalization_factor_(1.f / max_incr_per_pixel),
        split_polarity_(split_polarity),
        num_utbins_(num_utbins),
        clip_value_after_normalization_(clip_value_after_normalization),
        num_utbins_over_delta_t_(static_cast<float>(num_utbins) / delta_t),
        w_h_(network_input_width_ * network_input_height_) {
        assert(num_utbins > 0);
        assert(max_incr_per_pixel > 0.f);
        assert(input_height_scaling_ > 0.f);
        assert(input_width_scaling_ > 0.f);
        assert(clip_value_after_normalization_ >= 0.f);
        normalization_factor_scaled_ = normalization_factor_ * input_height_scaling_ * input_width_scaling_;

        num_polarities_       = split_polarity_ ? 2 : 1;
        network_num_channels_ = num_polarities_ * num_utbins_;
        w_h_p_                = network_input_width_ * network_input_height_ * num_polarities_;
    }

private:
    inline void set_value(float *buff, const std::size_t buff_size, const int bin, const int p, const int x,
                          const int y, const float val) const {
        const int idx = x + (y * network_input_width_) + p * (w_h_) + bin * (w_h_p_);
        assert(idx >= 0);
        assert(idx < static_cast<int>(buff_size));
        if (clip_value_after_normalization_ != 0.f) {
            buff[idx] =
                std::max(-clip_value_after_normalization_, std::min(clip_value_after_normalization_, buff[idx] + val));
        } else {
            buff[idx] += val;
        }
    }

    virtual void compute(const timestamp cur_frame_start_ts, const EventCD *begin, const EventCD *end, float *buff,
                         const std::size_t buff_size) const {
        assert(buff_size == network_input_height_ * network_input_width_ * network_num_channels_);
        assert(normalization_factor_scaled_ != 0.f);
        for (auto it = begin; it != end; ++it) {
            auto &ev = *it;
            assert((ev.p == 0) || (ev.p == 1));
            assert(ev.t >= cur_frame_start_ts);
            assert(ev.t < cur_frame_start_ts + delta_t_);
            assert(ev.x >= 0);
            assert(ev.x < network_input_width_);
            assert(ev.y >= 0);
            assert(ev.y < network_input_height_);
            int x = ev.x;
            int y = ev.y;

            float ti_star  = ((ev.t - cur_frame_start_ts) * num_utbins_over_delta_t_) - 0.5f;
            const int lbin = floor(ti_star);
            const int rbin = lbin + 1;

            float left_value  = std::max(0.f, 1.f - std::abs(lbin - ti_star));
            float right_value = 1.f - left_value;

            const int p = split_polarity_ ? ev.p : 0;
            if (!split_polarity_) {
                const int pol = ev.p ? ev.p : -1;
                left_value *= pol;
                right_value *= pol;
            }

            if ((lbin >= 0) && (lbin < num_utbins_)) {
                set_value(buff, buff_size, lbin, p, x, y, left_value * normalization_factor_scaled_);
            }
            if (rbin < num_utbins_) {
                set_value(buff, buff_size, rbin, p, x, y, right_value * normalization_factor_scaled_);
            }
        }
    }

    float increment_;
    float normalization_factor_;
    float normalization_factor_scaled_;
    bool split_polarity_;
    int num_utbins_;
    float clip_value_after_normalization_;
    int num_polarities_;
    float num_utbins_over_delta_t_;
    const int w_h_; // network_input_width * network_input_height
    int w_h_p_;     // network_input_width * network_input_height * num_polarities
};

} // namespace Metavision

#endif // METAVISION_SDK_ML_DETAIL_CD_PROCESSING_EVENT_CUBE_H
