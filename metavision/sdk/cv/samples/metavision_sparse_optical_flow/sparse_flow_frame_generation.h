/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_SPARSE_FLOW_FRAME_GENERATION_H
#define METAVISION_SDK_CV_SPARSE_FLOW_FRAME_GENERATION_H

#include <opencv2/opencv.hpp>

#include <metavision/sdk/cv/algorithms/sparse_flow_frame_generator_algorithm.h>
#include <metavision/sdk/base/utils/object_pool.h>

namespace Metavision {

struct SparseFlowFrameGeneration {
    using OutputCb    = std::function<void(timestamp, const cv::Mat &)>;
    using EventBuffer = std::vector<EventCD>;

    /// @brief Constructor
    /// @param width Width of the generated frame (in pixels)
    /// @param height Height of the generated frame (in pixels)
    /// @param fps The frame rate of the generated sequence of frames
    SparseFlowFrameGeneration(int width, int height, int fps) : width_(width), height_(height) {
        display_accumulation_time_us_ = 10000;
        bg_color_                     = cv::Vec3b(52, 37, 30);
        on_color_                     = cv::Vec3b(236, 223, 216);
        off_color_                    = cv::Vec3b(201, 126, 64);
        colored_                      = true;
        frame_period_                 = static_cast<timestamp>(1.e6 / fps + 0.5);
        next_process_ts_              = frame_period_;
        cd_frame_updated_             = false;
        frame_                        = cv::Mat(width, height, 3);
    }

    void set_output_callback(const OutputCb &output_cb) {
        on_new_frame_ = output_cb;
    }

    void process_cd_events(std::vector<EventCD> &cd_events) {
        try {
            if (!cd_events.empty()) {
                auto out_buffer = event_buffer_pool_.acquire();
                std::swap(*out_buffer, cd_events);
                cd_buffers_.emplace(out_buffer);
                timestamp last_cd_ts = -1, last_flow_ts = -1;
                if (!cd_buffers_.empty() && !cd_buffers_.back()->empty())
                    last_cd_ts = cd_buffers_.back()->back().t;
                if (!flow_buffers_.empty() && !flow_buffers_.back().empty())
                    last_flow_ts = flow_buffers_.back().back().t;
                while (last_cd_ts >= next_process_ts_ && last_flow_ts >= next_process_ts_) {
                    generate();
                }
            }
        } catch (boost::bad_any_cast &) {}
    }

    void process_flow_events(const std::vector<EventOpticalFlow> &flow_events) {
        try {
            if (!flow_events.empty()) {
                flow_buffers_.emplace(flow_events);
                timestamp last_cd_ts = -1, last_flow_ts = -1;
                if (!cd_buffers_.empty() && !cd_buffers_.back()->empty())
                    last_cd_ts = cd_buffers_.back()->back().t;
                if (!flow_buffers_.empty() && !flow_buffers_.back().empty())
                    last_flow_ts = flow_buffers_.back().back().t;
                while (last_cd_ts >= next_process_ts_ && last_flow_ts >= next_process_ts_) {
                    generate();
                }
            }
        } catch (boost::bad_any_cast &) {}
    }

private:
    void generate() {
        timestamp ts_end   = next_process_ts_;
        timestamp ts_begin = ts_end - display_accumulation_time_us_;
        update_frame_with_cd(ts_begin, ts_end, frame_);
        update_frame_with_flow(ts_begin, ts_end, frame_);
        if (cd_frame_updated_) {
            // ignore flow if cd frame has not been updated
            on_new_frame_(next_process_ts_, frame_);
            cd_frame_updated_ = false;
        }
        next_process_ts_ += frame_period_;
    }

    void update_frame_with_cd(timestamp ts_begin, timestamp ts_end, cv::Mat &frame) {
        // update frame type if needed
        frame.create(height_, width_, colored_ ? CV_8UC3 : CV_8U);
        frame.setTo(bg_color_);

        while (!cd_buffers_.empty()) {
            auto buffer = cd_buffers_.front();
            if (!buffer->empty()) {
                auto it_begin = std::lower_bound(std::begin(*buffer), std::end(*buffer), EventCD(0, 0, 0, ts_begin),
                                                 [](const auto &ev1, const auto &ev2) { return ev1.t < ev2.t; });
                auto it_end   = std::lower_bound(std::begin(*buffer), std::end(*buffer), EventCD(0, 0, 0, ts_end),
                                                 [](const auto &ev1, const auto &ev2) { return ev1.t < ev2.t; });
                if (it_begin != it_end) {
                    if (colored_) {
                        for (auto it = it_begin; it != it_end; ++it) {
                            frame.at<cv::Vec3b>(it->y, it->x) = it->p ? on_color_ : off_color_;
                        }
                    } else {
                        for (auto it = it_begin; it != it_end; ++it) {
                            frame.at<uint8_t>(it->y, it->x) = it->p ? on_color_[0] : off_color_[0];
                        }
                    }
                    cd_frame_updated_ = true;
                }
                if (it_end != std::end(*buffer))
                    break;
            }
            cd_buffers_.pop();
        }
    }

    void update_frame_with_flow(timestamp ts_begin, timestamp ts_end, cv::Mat &frame) {
        while (!flow_buffers_.empty()) {
            auto buffer = flow_buffers_.front();
            if (!buffer.empty()) {
                auto it_begin = std::lower_bound(std::begin(buffer), std::end(buffer), ts_begin,
                                                 [](const auto &ev, timestamp t) { return ev.t < t; });
                auto it_end   = std::lower_bound(std::begin(buffer), std::end(buffer), ts_end,
                                                 [](const auto &ev, timestamp t) { return ev.t < t; });
                if (it_begin != it_end) {
                    algo_.add_flow_for_frame_update(it_begin, it_end);
                }
                if (it_end != std::end(buffer))
                    break;
            }
            flow_buffers_.pop();
        }
        algo_.clear_ids();
        algo_.update_frame_with_flow(frame);
    }

    // Image to display
    int width_, height_;
    timestamp frame_period_ = -1;
    cv::Scalar bg_color_;
    cv::Vec3b on_color_, off_color_;
    bool colored_;
    bool cd_frame_updated_;
    cv::Mat frame_;
    OutputCb on_new_frame_;

    // Time interval to display events
    uint32_t display_accumulation_time_us_ = 5000;

    // Next processing timestamp according to frame_period value
    timestamp next_process_ts_ = 0;

    SharedObjectPool<EventBuffer> event_buffer_pool_;
    std::queue<SharedObjectPool<EventBuffer>::ptr_type> cd_buffers_;
    std::queue<std::vector<EventOpticalFlow>> flow_buffers_;
    SparseFlowFrameGeneratorAlgorithm algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_SPARSE_FLOW_FRAME_GENERATION_H