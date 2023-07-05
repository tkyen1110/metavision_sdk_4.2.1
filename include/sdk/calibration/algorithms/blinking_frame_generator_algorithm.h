/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_BLINKING_FRAME_GENERATOR_ALGORITHM_H
#define METAVISION_SDK_CALIBRATION_BLINKING_FRAME_GENERATOR_ALGORITHM_H

#include <opencv2/core/core.hpp>
#include <functional>
#include <memory>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/calibration/configs/blinking_frame_generator_algorithm_config.h"

namespace Metavision {
class BlinkingFrameGeneratorAlgorithmInternal;

/// @brief Class that generates a frame from blinking events
///
/// It accumulates events and keeps pixels which were activated with both polarities during the
/// accumulating period, if enough of them are found. Outputs a binary frame representing blinking pixels (0 or 255).
class BlinkingFrameGeneratorAlgorithm {
public:
    /// Type for callbacks called after each asynchronous process
    using OutputCb = std::function<void(timestamp, cv::Mat &)>;

    /// @brief Constructor
    /// @param sensor_width Sensor's width
    /// @param sensor_height Sensor's height
    /// @param config Blinking detector configuration
    BlinkingFrameGeneratorAlgorithm(int sensor_width, int sensor_height,
                                    const BlinkingFrameGeneratorAlgorithmConfig &config);

    ~BlinkingFrameGeneratorAlgorithm();

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end);

    /// @brief Sets a callback to get the mask
    /// @note The generated image will be passed to the callback as a non constant reference, meaning that the
    /// user is free to copy it or swap it. In case of a swap with a non initialized image, it will be automatically
    /// initialized
    void set_output_callback(const OutputCb &output_cb);

private:
    std::unique_ptr<BlinkingFrameGeneratorAlgorithmInternal> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CALIBRATION_BLINKING_FRAME_GENERATOR_ALGORITHM_H
