/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_DETAIL_CD_PROCESSING_HISTO_H
#define METAVISION_SDK_ML_DETAIL_CD_PROCESSING_HISTO_H

#include "metavision/sdk/ml/algorithms/cd_processing_algorithm.h"

namespace Metavision {

/// @brief Class used to compute the Histo CD processing
class CDProcessingHisto : public CDProcessing {
public:
    CDProcessingHisto() = default;

    /// @brief Constructor
    /// @param delta_t Delta time used to accumulate events inside the frame
    /// @param network_input_width Neural network input frame's width
    /// @param network_input_height Neural network input frame's height
    /// @param max_incr_per_pixel Maximum number of increments per pixel. This is used to normalize the contribution of
    /// each event
    /// @param clip_value_after_normalization Clipping value to apply after normalization (typically: 1.)
    /// @param event_input_width Sensor's width
    /// @param event_input_height Sensor's height
    /// @param use_CHW Boolean to define frame dimension order,
    ///     True if the fields' frame order is (Channel, Height, Width)
    CDProcessingHisto(timestamp delta_t, int network_input_width, int network_input_height, float max_incr_per_pixel,
                      float clip_value_after_normalization, int event_input_width = 0, int event_input_height = 0,
                      bool use_CHW = true) :
        CDProcessing(delta_t, network_input_width, network_input_height, event_input_width, event_input_height,
                     use_CHW),
        clip_value_after_normalization_(clip_value_after_normalization) {
        network_num_channels_ = 2;
        if (max_incr_per_pixel == 0.f) {
            throw std::invalid_argument("max_incr_per_pixel can't be 0");
        }
        if (clip_value_after_normalization <= 0.f) {
            throw std::invalid_argument("clip_value_after_normalization must be > 0");
        }
        increment_ = 1.f / max_incr_per_pixel;
        assert(input_height_scaling_ > 0.f);
        assert(input_width_scaling_ > 0.f);
        // further normalize the increment_ if rescaling event coordinates to make up for adding more events per
        // histogram cell
        increment_ *= input_width_scaling_ * input_height_scaling_;
    }

private:
    virtual void compute(const timestamp cur_frame_start_ts, const EventCD *begin, const EventCD *end, float *buff,
                         std::size_t buff_size) const {
        assert(network_num_channels_ == 2);
        assert(buff_size == network_input_height_ * network_input_width_ * network_num_channels_);
        for (auto it = begin; it != end; ++it) {
            const auto &ev = *it;
            assert((ev.p == 0) || (ev.p == 1));
            assert(ev.t >= cur_frame_start_ts);
            assert(ev.t < cur_frame_start_ts + delta_t_);
            assert(ev.x >= 0);
            assert(ev.x < network_input_width_);
            assert(ev.y >= 0);
            assert(ev.y < network_input_height_);
            int idx = 0;

            if (use_CHW_) {
                idx = network_input_width_ * (network_input_height_ * ev.p + ev.y) + ev.x;
            } else {
                idx = network_num_channels_ * (network_input_width_ * ev.y + ev.x) + ev.p;
            }

            assert(idx < static_cast<int>(buff_size));
            buff[idx] = std::min(clip_value_after_normalization_, buff[idx] + increment_);
        }
    }

    float increment_;
    const float clip_value_after_normalization_;
};

} // namespace Metavision

#endif //  METAVISION_SDK_ML_DETAIL_CD_PROCESSING_HISTO_H
