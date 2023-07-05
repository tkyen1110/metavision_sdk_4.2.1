/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include "metavision/sdk/ml/algorithms/cd_processing_algorithm.h"

#ifndef METAVISION_SDK_ML_DETAIL_CD_PROCESSING_ALGORITHM_IMPL_H
#define METAVISION_SDK_ML_DETAIL_CD_PROCESSING_ALGORITHM_IMPL_H

namespace Metavision {

/// @brief Constructs a CDProcessing object to ease the neural network input frame
/// @param delta_t Delta time used to accumulate events inside the frame
/// @param network_input_width Neural network input frame's width
/// @param network_input_height Neural network input frame's height
/// @param num_channels Number of channel in the neural network input frame
/// @param event_input_width Sensor's width
/// @param event_input_height Sensor's height
/// @param use_CHW Boolean to define frame dimension order,
///     True if the fields's frame order is (Channel, Height, Width)
CDProcessing::CDProcessing(timestamp delta_t, int network_input_width, int network_input_height, int event_input_width,
                           int event_input_height, bool use_CHW) :
    delta_t_(delta_t),
    network_input_width_(network_input_width),
    network_input_height_(network_input_height),
    network_num_channels_(0),
    event_input_width_((event_input_width) ? event_input_width : network_input_width_),
    event_input_height_((event_input_height) ? event_input_height : network_input_height_),
    use_CHW_(use_CHW) {
    analyse_args(delta_t, network_input_width_, network_input_height_, event_input_width_, event_input_height_,
                 &cd_rescaling_type_);

    if (event_input_width_ != network_input_width_ || event_input_height_ != network_input_height_) {
        if (network_input_width_ > event_input_width_ || network_input_height_ > event_input_height_) {
            input_width_scaling_  = static_cast<float>(network_input_width - 1) / (event_input_width_ - 1);
            input_height_scaling_ = static_cast<float>(network_input_height_ - 1) / (event_input_width_ - 1);
        } else {
            input_width_scaling_  = static_cast<float>(network_input_width_) / event_input_width_;
            input_height_scaling_ = static_cast<float>(network_input_height_) / event_input_height_;
        }
    } else {
        input_width_scaling_  = 1.f;
        input_height_scaling_ = 1.f;
    }
}

/// @brief Updates the frame depending on the input events
/// @param cur_frame_start_ts starting timestamp of the current frame
/// @param begin Begin iterator
/// @param end End iterator
/// @param frame Pointer to the frame (input/output)
/// @param frame_size Input frame size
template<typename InputIt>
inline void CDProcessing::operator()(const timestamp cur_frame_start_ts, InputIt begin, InputIt end, float *frame,
                                     int frame_size) const {
    if (begin == end) {
        return;
    }
    const EventCD *begin_ptr = nullptr;
    const EventCD *end_ptr   = nullptr;
    switch (cd_rescaling_type_) {
    case CDProcessingRescalingType::None:
        begin_ptr = &(*begin);
        end_ptr   = begin_ptr + std::distance(begin, end);
        break;
    case CDProcessingRescalingType::Downscaling:
        vect_events_rescaled_.resize(std::distance(begin, end));
        std::copy(begin, end, vect_events_rescaled_.begin());
        for (auto it = vect_events_rescaled_.begin(); it != vect_events_rescaled_.end(); ++it) {
            it->x = static_cast<int>(it->x * input_width_scaling_);
            it->y = static_cast<int>(it->y * input_height_scaling_);
        }
        assert(!vect_events_rescaled_.empty());
        begin_ptr = vect_events_rescaled_.data();
        end_ptr   = begin_ptr + vect_events_rescaled_.size();
        break;
    case CDProcessingRescalingType::Upscaling:
        vect_events_rescaled_.resize(std::distance(begin, end));
        std::copy(begin, end, vect_events_rescaled_.begin());
        for (auto it = vect_events_rescaled_.begin(); it != vect_events_rescaled_.end(); ++it) {
            it->x = static_cast<int>(it->x * input_width_scaling_ + 0.5f);
            it->y = static_cast<int>(it->y * input_height_scaling_ + 0.5f);
        }
        begin_ptr = vect_events_rescaled_.data();
        end_ptr   = begin_ptr + vect_events_rescaled_.size();
        break;
    default:
        throw std::runtime_error("Wrong rescaling_type");
    }
    compute(cur_frame_start_ts, begin_ptr, end_ptr, frame, frame_size);
}

/// @brief Check the arguments are compatible and valid for the computation
/// @param delta_t Delta time used to accumulate events inside the frame
/// @param network_input_width Neural network input frame's width
/// @param network_input_height Neural network input frame's height
/// @param event_input_width Sensor's width
/// @param event_input_height Sensor's height
/// @param rescaling_type_ptr Pointer on memory to store the required rescaling method
void CDProcessing::analyse_args(timestamp delta_t, int network_input_width, int network_input_height,
                                int event_input_width, int event_input_height,
                                CDProcessingRescalingType *rescaling_type_ptr) {
    if (delta_t <= 0) {
        std::ostringstream oss;
        oss << "CDProcessing : delta_t must be strictly positive: " << delta_t << std::endl;
        throw std::invalid_argument(oss.str());
    }

    if ((network_input_width <= 1) || (network_input_height <= 1)) {
        std::ostringstream oss;
        oss << "CDProcessing : invalid value for network input frame (width and height must be > 1): ";
        oss << network_input_width << "x" << network_input_height << std::endl;
        throw std::invalid_argument(oss.str());
    }

    if ((event_input_width <= 1) || (event_input_height <= 1)) {
        std::ostringstream oss;
        oss << "CDProcessing : invalid value for event input frame (width and height must be > 1): ";
        oss << event_input_width << "x" << event_input_height << std::endl;
        throw std::invalid_argument(oss.str());
    }

    if ((network_input_width > event_input_width && network_input_height <= event_input_height) ||
        (network_input_width <= event_input_width && network_input_height > event_input_height)) {
        std::ostringstream oss;
        oss << "CDProcessing : unsupported operations, scaling is different in each dimension. ";
        oss << "Please check the dimensions of the event frame ";
        oss << "(" << event_input_width << "x" << event_input_height << ") and network input frame ";
        oss << "(" << network_input_width << "x" << network_input_height << ")" << std::endl;
        throw std::logic_error(oss.str());
    }

    if (rescaling_type_ptr) {
        CDProcessingRescalingType rescaling_type = CDProcessingRescalingType::None;
        if (event_input_width != network_input_width || event_input_height != network_input_height) {
            if (network_input_width > event_input_width || network_input_height > event_input_height) {
                rescaling_type = CDProcessingRescalingType::Upscaling;
            } else {
                rescaling_type = CDProcessingRescalingType::Downscaling;
            }
        } else {
            rescaling_type = CDProcessingRescalingType::None;
        }
        *rescaling_type_ptr = rescaling_type;
    }
}
} // namespace Metavision

#endif // METAVISION_SDK_ML_DETAIL_CD_PROCESSING_ALGORITHM_IMPL_H
