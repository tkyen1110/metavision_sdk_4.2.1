/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_ALGORITHMS_CD_PROCESSING_ALGORITHM_H
#define METAVISION_SDK_ML_ALGORITHMS_CD_PROCESSING_ALGORITHM_H

#include <string>
#include <cassert>
#include <sstream>
#include <map>
#include <memory>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "metavision/sdk/base/events/event_cd.h"

#include "detail/cd_processing_rescaling_type.h"

namespace Metavision {

/// Frame input format
using Frame_t = std::vector<float>;

/// @brief Processes CD event to compute neural network input frame (3 dimensional tensor)
///
/// This is the base class. It handles the rescaling of the events if necessary.
/// It also provides accessors to get the shape of the output tensor.
/// Derived class implement the computation.
/// Calling operator() on this base class triggers the computation
class CDProcessing {
public:
    CDProcessingRescalingType cd_rescaling_type_;

    /// @brief Constructs a CDProcessing object to ease the neural network input frame
    /// @param delta_t Delta time used to accumulate events inside the frame
    /// @param network_input_width Neural network input frame's width
    /// @param network_input_height Neural network input frame's height
    /// @param event_input_width Sensor's width
    /// @param event_input_height Sensor's height
    /// @param use_CHW Boolean to define frame dimension order,
    ///     True if the fields' frame order is (Channel, Height, Width)
    inline CDProcessing(timestamp delta_t, int network_input_width, int network_input_height, int event_input_width = 0,
                        int event_input_height = 0, bool use_CHW = true);

    /// @brief Gets the frame size
    /// @return the frame size in pixel (height * width * channels)
    size_t get_frame_size() const {
        return network_input_width_ * network_input_height_ * network_num_channels_;
    };

    /// @brief Gets the network's input frame's width
    /// @return Network input frame's width
    size_t get_frame_width() const {
        return network_input_width_;
    }

    /// @brief Gets the network's input frame's height
    /// @return Network input frame's height
    size_t get_frame_height() const {
        return network_input_height_;
    }

    /// @brief Gets the number of channel in network input frame
    /// @return Number of channel in network input frame
    size_t get_frame_channels() const {
        return network_num_channels_;
    }

    /// @brief Checks the tensor's dimension order
    /// @return true if the dimension order is (channel, height, width)
    bool is_CHW() const {
        return use_CHW_;
    }

    /// @brief Gets the shape of the frame (3 dim, either CHW or HWC)
    /// @return a vector of sizes
    std::vector<size_t> get_frame_shape() const {
        std::vector<size_t> shape;
        if (use_CHW_) {
            shape.push_back(network_num_channels_);
            shape.push_back(network_input_height_);
            shape.push_back(network_input_width_);
        } else {
            shape.push_back(network_input_height_);
            shape.push_back(network_input_width_);
            shape.push_back(network_num_channels_);
        }
        return shape;
    }

    /// @brief Updates the frame depending on the input events
    /// @tparam InputIt type of input iterator (either a container iterator or raw pointer to EventCD)
    /// @param cur_frame_start_ts starting timestamp of the current frame
    /// @param begin Begin iterator
    /// @param end End iterator
    /// @param frame Pointer to the frame (input/output)
    /// @param frame_size Input frame size
    template<typename InputIt>
    inline void operator()(const timestamp cur_frame_start_ts, InputIt begin, InputIt end, float *frame,
                           int frame_size) const;

protected:
    timestamp delta_t_;
    const int network_input_width_, network_input_height_;
    int network_num_channels_;
    float input_width_scaling_, input_height_scaling_;

private:
    virtual void compute(const timestamp cur_frame_start_ts, const EventCD *begin, const EventCD *end, float *buff,
                         std::size_t buff_size) const = 0;

    /// @brief Checks the arguments are compatible and valid for the computation
    /// @param delta_t Delta time used to accumulate events inside the frame
    /// @param network_input_width Neural network input frame's width
    /// @param network_input_height Neural network input frame's height
    /// @param event_input_width Sensor's width
    /// @param event_input_height Sensor's height
    /// @param rescaling_type_ptr Pointer on memory to store the required rescaling method
    inline void analyse_args(timestamp delta_t, int network_input_width, int network_input_height,
                             int event_input_width, int event_input_height,
                             CDProcessingRescalingType *rescaling_type_ptr = nullptr);

    const int event_input_width_, event_input_height_;
    mutable std::vector<EventCD> vect_events_rescaled_; // temporary buffer used to compute the scaling

protected:
    bool use_CHW_; // True by default. If false, it will create HWC instead
};

} // namespace Metavision

#include "metavision/sdk/ml/algorithms/detail/cd_processing_algorithm_impl.h"

#endif // METAVISION_SDK_ML_ALGORITHMS_CD_PROCESSING_ALGORITHM_H
