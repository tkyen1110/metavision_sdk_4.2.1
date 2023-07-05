/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_PATTERN_BLINKER_HELPER_H
#define METAVISION_SDK_CALIBRATION_PATTERN_BLINKER_HELPER_H

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/object_pool.h>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/calibration/utils/pattern_blinker.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <atomic>

namespace Metavision {

/// @brief Class that produces a blinking pattern at a fixed frequency
///
///   - Input : None
///   - Output: Window displaying timestamped frame (Blinking Pattern)
class BlinkingPatternGenerator {
public:
    /// @brief Construct a new Pattern Blinker Generator object
    ///
    /// @param pattern_image_path Image to blink
    /// @param display_height Height of blinking window in pixels
    /// @param refresh_period_us Period between showing a blank image and the input image in microseconds
    BlinkingPatternGenerator(const std::string &pattern_image_path, int display_height,
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
        window_ = std::make_unique<Metavision::Window>("Blinking Pattern", display_width, display_height,
                                                       Metavision::BaseWindow::RenderMode::GRAY);
        window_->set_keyboard_callback(
            [&](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
                if (action == Metavision::UIAction::RELEASE &&
                    (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                    window_->set_close_flag();
                }
            });
    }

    void start() {
        rendering_thread_ = std::thread(&BlinkingPatternGenerator::render_loop, this);
    }

    void end() {
        stop_rendering_loop_ = true;
        if (rendering_thread_.joinable())
            rendering_thread_.join();
    }
    bool should_close() {
        if (window_->should_close()) {
            return true;
        }
        return false;
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
                cv::resize(tmp_img_, frame_, display_size_);
                window_->show(frame_);
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
    std::unique_ptr<Metavision::Window> window_;
    cv::Mat frame_;

    cv::Size display_size_;
    std::unique_ptr<Metavision::PatternBlinker> blinker_;
    cv::Mat tmp_img_;
};
} // namespace Metavision
#endif