/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_BLINKING_DOTS_GRID_DETECTOR_ALGORITHM_H
#define METAVISION_SDK_CALIBRATION_BLINKING_DOTS_GRID_DETECTOR_ALGORITHM_H

#include <tuple>
#include <memory>
#include <functional>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/calibration/configs/blinking_dots_grid_detector_algorithm_config.h"

namespace Metavision {

class BlinkingDotsGridDetectorAlgorithmInternal;

/// @brief Class that detects a grid of blinking dots with specific frequencies, for example the Prophesee calibration
/// shield.
class BlinkingDotsGridDetectorAlgorithm {
public:
    /// Type for callback.
    using OutputCb = std::function<void(timestamp, cv::Mat &, std::vector<cv::Point2f> &)>;

    /// @brief Constructor.
    /// @param sensor_width Sensor's width.
    /// @param sensor_height Sensor's height.
    /// @param config Configuration parameters, see @ref BlinkingDotsGridDetectorAlgorithmConfig.
    BlinkingDotsGridDetectorAlgorithm(int sensor_width, int sensor_height,
                                      const BlinkingDotsGridDetectorAlgorithmConfig &config);

    /// @brief Destructor.
    ~BlinkingDotsGridDetectorAlgorithm();

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to last input event
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end);

    /// @brief Sets a callback to get the detection results.
    /// The callback provides:
    /// * Timestamp of grid detection.
    /// * Image showing the location of the events used to detect the grid.
    /// * std::vector with the points in the grid (empty if the grid was not correctly detected).
    /// @note The generated image and the vector will be passed to the callback as non constant
    /// references. The user is free to copy or swap them.
    /// @param output_cb Function to be set as callback.
    void set_output_callback(const OutputCb &output_cb);

private:
    std::unique_ptr<BlinkingDotsGridDetectorAlgorithmInternal> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CALIBRATION_BLINKING_DOTS_GRID_DETECTOR_ALGORITHM_H
