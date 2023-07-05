/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// Tool for camera focusing by means of a blinking pattern, using Metavision Calibration SDK.

#include <functional>
#include <regex>
#include <chrono>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/utils/frame_composer.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/calibration/algorithms/blinking_frame_generator_algorithm.h>
#include <metavision/sdk/calibration/utils/pattern_blinker.h>
#include <metavision/sdk/calibration/configs/dft_high_freq_scorer_algorithm_config.h>
#include <metavision/sdk/calibration/algorithms/dft_high_freq_scorer_algorithm.h>
#include <metavision/sdk/calibration/algorithms/blinking_frame_generator_algorithm.h>
#include <metavision/sdk/calibration/utils/pattern_blinker.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/window.h>

#include "blinking_pattern_generator.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

/// @brief Class that computes and produces the Discrete Fourier Transform of an image with a score depending on the
/// proportion of high frequencies.
///
/// @note It produces a frame with the High Frequency score written on it
///   - Input : timestamped frame (Blinking Chessboard)   : Frame
///   - Output: timestamped frame (Score, White on Black) : Frame
class DftHighFreqScorer {
public:
    DftHighFreqScorer(int width, int height, const Metavision::DftHighFreqScorerAlgorithmConfig &dft_config,
                      unsigned int header_score_width, unsigned int header_score_height) :
        header_score_width_(header_score_width), header_score_height_(header_score_height) {
        high_freq_scorer_ = std::make_unique<Metavision::DftHighFreqScorerAlgorithm>(width, height, dft_config);
    }

    void process_dft(Metavision::timestamp ts, cv::Mat &in_frame, cv::Mat &score_frame) {
        float output_score;
        high_freq_scorer_->process_frame(ts, in_frame, output_score);
        score_frame.create(header_score_height_, header_score_width_, CV_8UC3);
        score_frame.setTo(0);
        const std::string score_str = std::to_string(100 * output_score);
        const cv::Size str_size     = cv::getTextSize(score_str, cv::FONT_HERSHEY_SIMPLEX, 1, 1, 0);
        cv::putText(score_frame, score_str,
                    cv::Point((score_frame.cols - str_size.width) / 2, (score_frame.rows + str_size.height) / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    }

private:
    std::unique_ptr<Metavision::DftHighFreqScorerAlgorithm> high_freq_scorer_;
    const int header_score_width_;
    const int header_score_height_;
};

// Application's parameters
struct Config {
    // Input/Output parameters
    std::string raw_file_path_;

    // Blinking frame generator algorithm's parameters
    Metavision::timestamp accumulation_time_;
    int min_num_blinking_pixels_;
    float blinking_pixels_ratios_on_;
    float blinking_pixels_ratios_off_;

    // Discrete Fourier Transform's parameters
    Metavision::timestamp refresh_period_us_;
    bool use_inverted_gray_ = false;

    // Pattern Blinker's parameters
    std::string pattern_image_path_;
    int pattern_blinker_height_;
    Metavision::timestamp pattern_blinker_refresh_period_us_;
};

bool get_configuration(int argc, char *argv[], Config &config) {
    const std::string program_desc(
        "Tool for camera focusing by means of a blinking pattern, using Metavision Calibration SDK.\n");

    po::options_description options_desc;
    po::options_description base_options("Base options");
    // clang-format off
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-raw-file,i", po::value<std::string>(&config.raw_file_path_), "Path to input RAW file. If not specified, the camera live stream is used.")
        ;
    // clang-format on

    po::options_description blinking_frame_generator_options("Blinking Frame Generator options");
    // clang-format off
    blinking_frame_generator_options.add_options()
        ("accumulation-time,a", po::value<Metavision::timestamp>(&config.accumulation_time_)->default_value(2e5), "Window of time during which events are considered to detect if a pixel is blinking.")
        ("min-blink-pix,m", po::value<int>(&config.min_num_blinking_pixels_)->default_value(0), "Minimum number of pixels needed to be detected before outputting a frame.")
        ("ratio-on",        po::value<float>(&config.blinking_pixels_ratios_on_)->default_value(1.0f), "The acceptable ratio of pixels that received only positive events over the number of pixels that received both during the accumulation window.")
        ("ratio-off",       po::value<float>(&config.blinking_pixels_ratios_off_)->default_value(1.0f),  "The acceptable ratio of pixels that received only negative events over the number of pixels that received both during the accumulation window.")
        ;
    // clang-format on

    po::options_description dft_options("Discrete Fourier Transform options");
    // clang-format off
    dft_options.add_options()
        ("dft-refresh", po::value<Metavision::timestamp >(&config.refresh_period_us_)->default_value(1e4), "Time period between two consecutive process (skip the blinking frames that are too close in time to the last one processed).")
        ("invert-gray", po::bool_switch(&config.use_inverted_gray_), "Invert the gray levels so that white becomes black (and conversely).")
        ;
    // clang-format on

    po::options_description pattern_blinker_options("Pattern Blinker options");
    // clang-format off
    pattern_blinker_options.add_options()
        ("pattern-image-path",  po::value<std::string>(&config.pattern_image_path_), "If a path to a pattern file is provided, display a blinking pattern on screen")
        ("pattern-blinker-height",  po::value<int>(&config.pattern_blinker_height_)->default_value(1080), "Height of the blinking pattern in pixels.")
        ("pattern-blinker-refresh", po::value<Metavision::timestamp>(&config.pattern_blinker_refresh_period_us_)->default_value(1e4), "Refresh period of the pattern blinker in us.")
        ;
    // clang-format on

    options_desc.add(base_options).add(blinking_frame_generator_options).add(dft_options).add(pattern_blinker_options);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return false;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return false;
    }

    return true;
}

int main(int argc, char *argv[]) {
    Config conf_;

    if (!get_configuration(argc, argv, conf_))
        return 1;

    const auto start = std::chrono::high_resolution_clock::now();

    Metavision::Camera camera;
    if (conf_.raw_file_path_.empty()) {
        try {
            camera = Metavision::Camera::from_first_available();
        } catch (const Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return 1;
        }
    } else {
        camera = Metavision::Camera::from_file(conf_.raw_file_path_,
                                               Metavision::FileConfigHints().real_time_playback(false));
    }

    const unsigned short width  = camera.geometry().width();
    const unsigned short height = camera.geometry().height();

    const unsigned int header_score_width                = width;
    const unsigned int header_score_height               = 50;
    const Metavision::timestamp event_buffer_duration_us = 100000;
    const int display_fps                                = 10;
    std::unique_ptr<Metavision::BlinkingPatternGenerator> pattern;

    // Data flow:
    //
    //  0 (Cam) -->-- 1 (STC) ---------->---------- 3 (Blink Frame Gen) -->---|
    //                |                             |                         |
    //                v                             v                         5 (Frame Composer) -->-- 6 (Display)
    //                |                             |                         |
    //                2 (Events Frame Gen)          4 (DFT) -------------->---|
    //                |                                                       |
    //                +------------------>-------------------------------->---|
    //
    //
    //  (optional)                  7 (Pattern Blinker) ----->----- 8 (Display)

    constexpr Metavision::timestamp stc_ths = 1e4;
    Metavision::SpatioTemporalContrastAlgorithm stc_algo(width, height, stc_ths, true);

    // Set Events Frame Generator
    Metavision::PeriodicFrameGenerationAlgorithm events_frame_gen(width, height, event_buffer_duration_us, display_fps);

    // Set Blinking Frame Generator
    Metavision::BlinkingFrameGeneratorAlgorithmConfig blinking_config(
        conf_.accumulation_time_, conf_.min_num_blinking_pixels_, conf_.blinking_pixels_ratios_on_,
        conf_.blinking_pixels_ratios_off_);
    Metavision::BlinkingFrameGeneratorAlgorithm blink_frame_gen(width, height, blinking_config);

    // Set Discrete Fourier Transformation
    Metavision::DftHighFreqScorerAlgorithmConfig dft_config(conf_.refresh_period_us_, conf_.use_inverted_gray_);
    DftHighFreqScorer high_freq_score(width, height, dft_config, header_score_width, header_score_height);

    std::vector<Metavision::EventCD> output;
    // Add callback that will pass the events to the STC filter. Events are then
    // passed to BlinkingFrameGenerator and EventsFrameGenerator
    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        stc_algo.process_events(begin, end, std::back_inserter(output));
        events_frame_gen.process_events(output.cbegin(), output.cend());
        blink_frame_gen.process_events(output.cbegin(), output.cend());
        output.clear();
    });

    // Set Frame Composer
    Metavision::FrameComposer frame_comp(cv::Vec3b(0, 0, 0));
    Metavision::FrameComposer::ResizingOptions resize_1(header_score_width, header_score_height, false);
    Metavision::FrameComposer::ResizingOptions resize_23(width, height, false);

    int subimage_1 = frame_comp.add_new_subimage_parameters(width + 10, 0, resize_1,
                                                            Metavision::FrameComposer::GrayToColorOptions());

    int subimage_2 = frame_comp.add_new_subimage_parameters(0, header_score_height + 10, resize_23,
                                                            Metavision::FrameComposer::GrayToColorOptions());
    int subimage_3 = frame_comp.add_new_subimage_parameters(width + 10, header_score_height + 10, resize_23,
                                                            Metavision::FrameComposer::GrayToColorOptions());

    const auto composed_width  = frame_comp.get_total_width();
    const auto composed_height = frame_comp.get_total_height();

    // Set display window
    Metavision::Window window("Raw events, Blinking events and High Frequency score ", composed_width, composed_height,
                              Metavision::BaseWindow::RenderMode::BGR);
    window.set_keyboard_callback(
        [&window](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
            if (action == Metavision::UIAction::RELEASE &&
                (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                window.set_close_flag();
            }
        });

    // Display the combined frame by adding callback for both Frame Generators
    events_frame_gen.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
        frame_comp.update_subimage(subimage_2, frame);
        window.show(frame_comp.get_full_image());
    });

    cv::Mat score_frame;
    blink_frame_gen.set_output_callback([&](Metavision::timestamp t, cv::Mat &frame) {
        frame_comp.update_subimage(subimage_3, frame);
        // Apply DFT
        high_freq_score.process_dft(t, frame, score_frame);
        frame_comp.update_subimage(subimage_1, score_frame);
        window.show(frame_comp.get_full_image());
    });

    // Display blinking pattern image, if enabled
    if (!conf_.pattern_image_path_.empty()) {
        try {
            pattern = std::make_unique<Metavision::BlinkingPatternGenerator>(
                conf_.pattern_image_path_, conf_.pattern_blinker_height_, conf_.pattern_blinker_refresh_period_us_);

            pattern->start();

        } catch (const std::runtime_error &e) {
            MV_LOG_ERROR() << e.what();
            return 1;
        }
    }

    camera.start();

    while (camera.is_running() && !window.should_close()) {
        Metavision::EventLoop::poll_and_dispatch(20);
        if (pattern && pattern->should_close()) {
            break;
        }
    }
    camera.stop();

    if (pattern) {
        pattern->end();
    }

    const auto end     = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    MV_LOG_INFO() << "Ran in" << static_cast<float>(elapsed.count()) / 1000.f << "s";

    return 0;
}
