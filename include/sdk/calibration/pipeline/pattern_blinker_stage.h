/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_PATTERN_BLINKER_STAGE_H
#define METAVISION_SDK_CALIBRATION_PATTERN_BLINKER_STAGE_H

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/object_pool.h>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/calibration/utils/pattern_blinker.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace Metavision {

/// @brief Stage that produces a blinking pattern at a fixed frequency
///
///   - Input : None
///   - Output: timestamped frame (Blinking Pattern) : FrameData
class PatternBlinkerStage : public Metavision::BaseStage {
public:
    using EventBuffer    = std::vector<Metavision::EventCD>;
    using EventBufferPtr = Metavision::SharedObjectPool<EventBuffer>::ptr_type;

    using FramePool = Metavision::SharedObjectPool<cv::Mat>;
    using FramePtr  = FramePool::ptr_type;
    using FrameData = std::pair<Metavision::timestamp, FramePtr>;

    /// @brief Construct a new Pattern Blinker Stage object
    ///
    /// @param pattern_image_path Image to blink
    /// @param display_height Height of blinking window
    /// @param refresh_period_us Period between showing a blank image and the input image in microseconds
    PatternBlinkerStage(const std::string &pattern_image_path, int display_height,
                        Metavision::timestamp refresh_period_us) {
        stop_rendering_loop_ = false;
        refresh_period_us_   = refresh_period_us;

        if (!boost::filesystem::is_regular_file(pattern_image_path))
            throw std::runtime_error("Pattern file not found at: " + pattern_image_path);

        const cv::Mat blink_pattern = cv::imread(pattern_image_path, cv::ImreadModes::IMREAD_GRAYSCALE);
        blinker_                    = std::make_unique<Metavision::PatternBlinker>(blink_pattern);

        const cv::Size img_size = blinker_->get_image_size();
        const int display_width = (img_size.width * display_height) / img_size.height;
        display_size_           = cv::Size(display_width, display_height);

        set_starting_callback([this]() { rendering_thread_ = std::thread(&PatternBlinkerStage::render_loop, this); });

        set_stopping_callback([this]() {
            stop_rendering_loop_ = true;
            if (rendering_thread_.joinable())
                rendering_thread_.join();
        });
    }

    cv::Size get_display_size() const {
        return display_size_;
    }

private:
    void render_loop() {
        Metavision::timestamp ts = 0;

        auto start = std::chrono::system_clock::now();

        while (!stop_rendering_loop_) {
            if (blinker_->update_blinking_image(ts, tmp_img_)) {
                auto output_frame_ptr = frame_pool_.acquire();
                cv::resize(tmp_img_, *output_frame_ptr, display_size_);
                produce(std::make_pair(ts, output_frame_ptr));
            }

            auto end        = std::chrono::system_clock::now();
            auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            start           = end;

            if (elapsed_us < refresh_period_us_)
                std::this_thread::sleep_for(std::chrono::microseconds(refresh_period_us_ - elapsed_us));

            ts += elapsed_us;
        }
    }

    std::atomic_bool stop_rendering_loop_;
    Metavision::timestamp refresh_period_us_;
    std::thread rendering_thread_;

    cv::Size display_size_;
    std::unique_ptr<Metavision::PatternBlinker> blinker_;
    FramePool frame_pool_;
    cv::Mat tmp_img_;
};
} // namespace Metavision
#endif