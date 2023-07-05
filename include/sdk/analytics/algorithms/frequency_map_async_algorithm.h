/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_FREQUENCY_MAP_ASYNC_ALGORITHM_H
#define METAVISION_SDK_ANALYTICS_FREQUENCY_MAP_ASYNC_ALGORITHM_H

#include <functional>
#include <memory>
#include <opencv2/core/mat.hpp>

#include "metavision/sdk/cv/configs/frequency_estimation_config.h"

namespace Metavision {

class FrequencyMapAsyncAlgorithmInternal;

/// @brief Class that estimates the pixel-wise frequency of vibrating objects using Metavision Vibration API
class FrequencyMapAsyncAlgorithm {
public:
    using OutputMap = cv::Mat1f;

    /// Type of the callback to access the period map
    using OutputCb = std::function<void(timestamp, OutputMap &)>;

    /// @brief Builds a new @ref FrequencyMapAsyncAlgorithm object
    /// @param width Sensor's width in pixels
    /// @param height Sensor's height in pixels
    /// @param frequency_config Frequency estimation configuration
    FrequencyMapAsyncAlgorithm(int width, int height, const FrequencyEstimationConfig &frequency_config);

    /// Default destructor
    ~FrequencyMapAsyncAlgorithm();

    /// @brief Sets a callback to get the output frequency map
    /// @param output_cb Callback to call
    void set_output_callback(const OutputCb &output_cb);

    /// @brief Sets the frequency at which the algorithm generates the frequency map
    /// @param freq Frequency at which the frequency map will be generated
    void set_update_frequency(const float freq);

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end);

private:
    std::unique_ptr<FrequencyMapAsyncAlgorithmInternal> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_FREQUENCY_MAP_ASYNC_ALGORITHM_H
