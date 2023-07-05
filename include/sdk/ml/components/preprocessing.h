/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_PREPROCESSING_H
#define METAVISION_SDK_ML_PREPROCESSING_H

#include <functional>
#include <sstream>

#include "metavision/sdk/core/algorithms/flip_x_algorithm.h"
#include "metavision/sdk/core/algorithms/flip_y_algorithm.h"
#include "metavision/sdk/core/algorithms/roi_filter_algorithm.h"
#include "metavision/sdk/cv/algorithms/transpose_events_algorithm.h"
#include "metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h"
#include "metavision/sdk/cv/algorithms/trail_filter_algorithm.h"
#include "metavision/sdk/base/events/event_cd.h"

namespace Metavision {

/// @brief Abstract object to provide generic method for processing events inside the slicer
template<typename EventType>
class PreprocessingBase {
public:
    /// @brief Function to transform events
    typedef std::function<void(const EventType *, const EventType *, std::vector<EventType> &)> PreProcessingEvent;

    /// @brief Returns a function to preprocess events
    /// @return function to be called on every events
    virtual PreProcessingEvent get_preprocessing_callback() = 0;
};

/// @brief Geometric events preprocessing
///
/// This class allows transforming input events by applying simple transformations like flip and transposition.
/// like flip and or transposition
template<typename EventType>
class GeometricPreprocessing : public PreprocessingBase<EventType> {
    using PreProcessingEvent = typename PreprocessingBase<EventType>::PreProcessingEvent;

public:
    /// @brief Builds GeometricPreprocessing object
    /// @param width Sensor's width
    /// @param height Sensor's width
    GeometricPreprocessing(int width, int height) : width_(width), height_(height) {}

    /// @brief Configures the preprocessing filter
    /// @param transpose If True, transposes events' X and Y coordinates
    /// @param flip_x Move origin to bottom of image
    /// @param flip_y Move origin to right of image
    void use_transpose_flipxy(bool transpose = false, bool flip_x = false, bool flip_y = false);

    /// @brief Returns the function to be called on every events
    /// @return Function to be called on every events
    virtual PreProcessingEvent get_preprocessing_callback() override final {
        if (!roi_filter_ && !flip_x_algo_ && !flip_y_algo_ && !transpose_events_algo_) {
            return nullptr;
        } else {
            return std::bind(&GeometricPreprocessing::process, this, std::placeholders::_1, std::placeholders::_2,
                             std::placeholders::_3);
        }
    };

    /// @brief Remove events outside of a region of interest (ROI)
    /// @param x X coordinate of ROI top left corner
    /// @param y Y coordinate of ROI top left corner
    /// @param w ROI's width
    /// @param h ROI's height
    void use_roi(int x, int y, int w, int h);

    /// @brief Processes input events
    /// @param begin First input event
    /// @param end Last input event
    /// @param tmp_buffer Vector to store transformed events
    void process(const EventType *begin, const EventType *end, std::vector<EventType> &tmp_buffer);

    /// @brief Gets width of output events
    /// @return Width of output events
    int get_width_after_preproc() const {
        if (transpose_events_algo_) {
            return height_;
        }
        return width_;
    }

    /// @brief Gets height of output events
    /// @return Height of output events
    int get_height_after_preproc() const {
        if (transpose_events_algo_) {
            return width_;
        }
        return height_;
    }

private:
    int width_;
    int height_;

    std::unique_ptr<RoiFilterAlgorithm> roi_filter_;
    std::unique_ptr<FlipXAlgorithm> flip_x_algo_;
    std::unique_ptr<FlipYAlgorithm> flip_y_algo_;
    std::unique_ptr<TransposeEventsAlgorithm> transpose_events_algo_;
};

/// @brief Class to pre-process events with a noise filter
template<typename EventType, typename NoiseFilterType>
class NoiseFilterPreprocessing : public PreprocessingBase<EventType> {
    using PreProcessingEvent = typename PreprocessingBase<EventType>::PreProcessingEvent;

public:
    /// @brief Builds a preprocessing object for noise filter
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @param noise_threshold Threshold to configure the noise filter
    NoiseFilterPreprocessing(int width, int height, timestamp noise_threshold) :
        noise_filter_(width, height, noise_threshold) {}

    /// @brief Applies noise filter on every events
    /// @param begin First event
    /// @param end Last event
    /// @param tmp_buffer Output vector of events
    void process(const EventType *begin, const EventType *end, std::vector<EventType> &tmp_buffer) {
        noise_filter_.process_events(begin, end, std::back_inserter(tmp_buffer));
    }

    /// @brief Returns the function to apply the noise filter
    /// @return function to be called on every events
    virtual PreProcessingEvent get_preprocessing_callback() override final {
        return std::bind(&NoiseFilterPreprocessing::process, this, std::placeholders::_1, std::placeholders::_2,
                         std::placeholders::_3);
    }

private:
    NoiseFilterType noise_filter_;
};

/// @brief Configures the preprocessing filter
/// @param transpose If True, transposes events' X and Y coordinates
/// @param flip_x Move origin to bottom of image
/// @param flip_y Move origin to right of image
template<typename EventType>
void GeometricPreprocessing<EventType>::use_transpose_flipxy(bool transpose, bool flip_x, bool flip_y) {
    int width_flip  = width_;
    int height_flip = height_;
    if (transpose) {
        transpose_events_algo_.reset(new TransposeEventsAlgorithm());
        width_flip  = height_;
        height_flip = width_;
    }

    if (flip_x) {
        flip_x_algo_.reset(new FlipXAlgorithm(width_flip - 1));
    }

    if (flip_y) {
        flip_y_algo_.reset(new FlipYAlgorithm(height_flip - 1));
    }
}

/// @brief Remove events outside of a region of interest (ROI)
/// @param x X coordinate of ROI top left corner
/// @param y Y coordinate of ROI top left corner
/// @param w ROI's width
/// @param h ROI's height
template<typename EventType>
void GeometricPreprocessing<EventType>::use_roi(int x, int y, int w, int h) {
    assert(x + w <= width_);
    assert(y + h <= height_);
    roi_filter_ = std::make_unique<RoiFilterAlgorithm>(x, y, x + w - 1, y + h - 1, false);
}

/// @brief Processes input events
/// @param begin First input event
/// @param end Last input event
/// @param tmp_buffer Vector to store transformed events
template<typename EventType>
void GeometricPreprocessing<EventType>::process(const EventType *begin, const EventType *end,
                                                std::vector<EventType> &tmp_buffer) {
    tmp_buffer.resize(std::distance(begin, end));

    if (!roi_filter_ && !flip_x_algo_ && !flip_y_algo_ && !transpose_events_algo_) {
        std::ostringstream oss;
        oss << "Warning: GeometricPreprocessing used, but no preproc is defined" << std::endl;
        throw std::runtime_error(oss.str());
    }

    auto ev_begin_ptr = const_cast<EventCD *>(begin);
    auto ev_end_ptr   = const_cast<EventCD *>(end);

    if (roi_filter_) {
        auto tmp_end = roi_filter_->process_events(ev_begin_ptr, ev_end_ptr, tmp_buffer.begin());
        ev_begin_ptr = &(tmp_buffer[0]);
        ev_end_ptr   = ev_begin_ptr + std::distance(tmp_buffer.begin(), tmp_end);
    }

    if (transpose_events_algo_) {
        auto tmp_end = transpose_events_algo_->process_events(ev_begin_ptr, ev_end_ptr, tmp_buffer.begin());
        ev_begin_ptr = &(tmp_buffer[0]);
        ev_end_ptr   = ev_begin_ptr + std::distance(tmp_buffer.begin(), tmp_end);
    }

    if (flip_x_algo_) {
        int nb_elem = std::distance(ev_begin_ptr, ev_end_ptr);
        flip_x_algo_->process_events(ev_begin_ptr, ev_end_ptr, tmp_buffer.begin());
        ev_begin_ptr = &(tmp_buffer[0]);
        ev_end_ptr   = ev_begin_ptr + nb_elem;
    }

    if (flip_y_algo_) {
        int nb_elem = std::distance(ev_begin_ptr, ev_end_ptr);
        flip_y_algo_->process_events(ev_begin_ptr, ev_end_ptr, tmp_buffer.begin());
        ev_begin_ptr = &(tmp_buffer[0]);
        ev_end_ptr   = ev_begin_ptr + nb_elem;
    }

    tmp_buffer.resize(std::distance(ev_begin_ptr, ev_end_ptr));
}

} // namespace Metavision

#endif // METAVISION_SDK_ML_PREPROCESSING_H
