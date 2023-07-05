/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_DENSE_FLOW_FRAME_GENERATOR_ALGORITHM_H
#define METAVISION_SDK_CV_DENSE_FLOW_FRAME_GENERATOR_ALGORITHM_H

#include <vector>
#include <opencv2/core.hpp>

#include "metavision/sdk/cv/events/event_optical_flow.h"

namespace Metavision {

/// @brief Algorithm used to generate visualization images of dense optical flow streams.
class DenseFlowFrameGeneratorAlgorithm {
public:
    /// @brief Policy for accumulating multiple flow events at a given pixel
    enum class AccumulationPolicy {
        Average,       ///< Computes the average flow from the observations at the pixel
        PeakMagnitude, ///< Keeps the highest magnitude flow amongst the observations at the pixel
        Last           ///< Keeps the most recent flow amongst the observations at the pixel
    };

    /// @brief Constructor
    /// @param width Sensor width
    /// @param height Sensor height
    /// @param maximum_flow_magnitude Scale for highest flow magnitude in the visualization.
    /// @param policy Method used to accumulate multiple flow values at the same pixel.
    DenseFlowFrameGeneratorAlgorithm(int width, int height, float maximum_flow_magnitude,
                                     AccumulationPolicy policy = AccumulationPolicy::Last);

    /// @brief Processes a buffer of flow events
    /// @tparam EventIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventOpticalFlow
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    /// @note Successive calls to process_events will accumulate data at each pixel until @ref generate or @ref reset
    /// is called.
    template<typename EventIt>
    void process_events(EventIt it_begin, EventIt it_end);

    /// @brief Generates a flow visualization frame
    /// @param frame Frame that will contain the flow visualization
    /// @param allocate Allocates the frame if true. Otherwise, the user must ensure the validity of the input frame.
    /// This is to be used when the data ptr must not change (external allocation, ROI over another cv::Mat, ...)
    /// @throw invalid_argument if the frame doesn't have the expected type and geometry
    void generate(cv::Mat &frame, bool allocate = true);

    /// @brief Generates a legend image for the flow visualization
    /// @param legend_frame Frame that will contain the flow visualization legend
    /// @param square_size Size of the generated image
    /// @param allocate Allocates the frame if true. Otherwise, the user must ensure the validity of the input frame.
    /// This is to be used when the data ptr must not change (external allocation, ROI over another cv::Mat, ...)
    /// @throw invalid_argument if the frame doesn't have the expected type
    void generate_legend_image(cv::Mat &legend_frame, int square_size = 0, bool allocate = true);

    /// @brief Resets the internal states
    void reset();

private:
    void flow_map_to_color_map(const cv::Mat_<float> &vx, const cv::Mat_<float> &vy, float max_flow_scale,
                               const cv::Mat &rgb);

    struct State {
        void reset();
        void accumulate(const EventOpticalFlow &ev_flow, AccumulationPolicy policy, float &vx, float &vy);
        std::uint32_t n = 0;
    };

    const int width_, height_;
    const float max_flow_magnitude_;
    const AccumulationPolicy policy_;
    cv::Mat_<float> vx_, vy_, mag_, ang_;
    cv::Mat hsv_, bgr_32f_;
    std::vector<State> states_;
};

} // namespace Metavision

#include "metavision/sdk/cv/algorithms/detail/dense_flow_frame_generator_algorithm_impl.h"

#endif // METAVISION_SDK_CV_DENSE_FLOW_FRAME_GENERATOR_ALGORITHM_H
